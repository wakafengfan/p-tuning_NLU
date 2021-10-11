import torch
import torch.nn as nn
import torch.optim
from bojone_snippets import *
from bojone_tokenizers import Tokenizer
from configuration.config import *
from opt import create_optimizer_and_scheduler
from transformers import AutoConfig, BertForMaskedLM

num_epoch = 10
batch_size = 32
maxlen = 128

dict_path = str(roberta_wwm_pt_path / "vocab.txt")
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


train_data = load_data(str(open_dataset_path / "sentiment" / "sentiment.train.data"))
valid_data = load_data(str(open_dataset_path / "sentiment" / "sentiment.valid.data"))
test_data = load_data(str(open_dataset_path / "sentiment" / "sentiment.test.data"))

# 模拟标注和非标注数据
train_frac = 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]

# 对应的任务描述
mask_idx = 5
desc = ['[unused%s]' % i for i in range(2, 10)]
desc.insert(mask_idx - 1, '[MASK]')
desc_ids = [tokenizer.token_to_id(t) for t in desc]
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if label != 2:
                token_ids = token_ids[:1] + desc_ids + token_ids[1:]
                segment_ids = [0] * len(desc_ids) + segment_ids
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long)
                batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long)
                batch_output_ids = torch.tensor(sequence_padding(batch_output_ids), dtype=torch.long)
                yield batch_token_ids, batch_segment_ids, batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained(roberta_wwm_pt_path / "config.json")


class PtuningModel(nn.Module):
    def __init__(self):
        super(PtuningModel, self).__init__()
        self.model = BertForMaskedLM.from_pretrained(roberta_wwm_pt_path, config=config)

    def forward(self, input_ids):
        input_embed = self.model.bert.embeddings(input_ids)

        att_mask = input_ids != tokenizer._token_pad_id

        output = self.model(inputs_embeds=input_embed.to(device),
                            attention_mask=att_mask.to(device),
                            output_hidden_states=True)

        return output.logits




tuning_model = PtuningModel()
tuning_model.to(device)
optimizer, lr_scheduler = create_optimizer_and_scheduler(tuning_model, lr=5e-5,
                                                         num_training_steps=1500)

for pn, p in tuning_model.named_parameters():
    if p.requires_grad:
        print(pn)

loss_func = nn.CrossEntropyLoss(reduction="none")


def compute_loss(y_pred, y_true):  # 只计算一个mask位置的loss
    y_mask = torch.not_equal(y_true, 0).float()
    loss = loss_func(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
    loss = (loss.view(y_pred.size(0), -1) * y_mask).sum() / y_mask.sum()

    return loss


best_val_acc = 0.
tuning_model.zero_grad()
for e in range(num_epoch):
    tuning_model.train()
    for step, batch in enumerate(train_generator.forfit()):
        batch = [_.to(device) for _ in batch]
        input_ids, seg_ids, label_ids = batch
        logits = tuning_model(input_ids)
        loss = compute_loss(logits, label_ids)

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step % 100 == 0 and step != 0:
            logger.info(f"epoch: {e} - step: {step} - loss: {loss.item()} - lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        if step == len(train_generator) * 50:
            break

    tuning_model.eval()
    total, right = 0., 0.
    for batch in tqdm(test_generator):
        input_ids, _, label_ids = batch
        with torch.no_grad():
            logits = tuning_model(input_ids.to(device))

        logits = logits.cpu()
        pred = logits[:, mask_idx, [neg_id, pos_id]].argmax(dim=1)
        y_true = (label_ids[:, mask_idx] == pos_id).long()
        total += input_ids.size(0)
        right += (y_true == pred).sum().item()

    val_acc = right / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    logger.info(f"epoch: {e} - acc: {val_acc:.6f} - best_test_acc: {best_val_acc}")

