import torch
import torch.nn as nn
from transformers import BertForMaskedLM, AutoConfig

from bojone_snippets import *
from bojone_tokenizers import Tokenizer
from configuration.config import *

num_epoch = 20
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
        batch_token_ids, batch_segment_ids, batch_output_ids, batch_pseudo_mask = [], [], [], []
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

            #################################
            batch_pseudo_mask.append([-1 if t_id in [2, 3, 4, 5, 6, 7, 8, 9] else 0 for t_id in token_ids])

            #################################

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long)
                batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long)
                batch_output_ids = torch.tensor(sequence_padding(batch_output_ids), dtype=torch.long)

                batch_pseudo_mask = torch.tensor(sequence_padding(batch_pseudo_mask), dtype=torch.long)

                yield batch_token_ids, batch_segment_ids, batch_output_ids, batch_pseudo_mask
                batch_token_ids, batch_segment_ids, batch_output_ids, batch_pseudo_mask = [], [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


# ========================================= data process end ===================================================


class PromptEncoder(nn.Module):
    def __init__(self, bert_word_embeddings, device):
        super(PromptEncoder, self).__init__()

        self.device = device
        self.seq_indices = torch.tensor(range(0, len(range(2, 10)))).to(self.device)
        self.init_weight = torch.index_select(bert_word_embeddings, dim=0, index=torch.tensor(range(2, 10)))

        self.prompt_embeddings = nn.Embedding(embedding_dim=bert_word_embeddings.size(1),
                                              num_embeddings=8,
                                              _weight=self.init_weight)

        self.hidden_size = bert_word_embeddings.size(1)
        self.lstm_head = nn.LSTM(input_size=self.hidden_size,
                                 hidden_size=self.hidden_size // 2,
                                 num_layers=2,
                                 dropout=0.0,
                                 bidirectional=True,
                                 batch_first=True
                                 )
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print(f"init prompt encoder ...")

    def forward(self):
        input_embeds = self.prompt_embeddings(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class PTuningModel(nn.Module):
    def __init__(self, device):
        super(PTuningModel, self).__init__()
        self.device = device
        self.model = BertForMaskedLM.from_pretrained(roberta_wwm_pt_path,
                                                     config=AutoConfig.from_pretrained(roberta_wwm_pt_path))
        for pn, p in self.model.named_parameters():
            p.requires_grad = False

        self.embeddings = self.model.get_input_embeddings()

        self.prompt_encoder = PromptEncoder(self.embeddings.weight, self.device)

        self.num_prompt = len(range(2, 10))

        self.pseudo_token_id = -1  # random place holder
        self.pad_token_id = 0

    def embed_input(self, input_ids, input_pseudo_mask):
        bz = input_ids.size(0)

        raw_embeds = self.embeddings(input_ids)
        replace_embeds = self.prompt_encoder()

        blocked_indices = (input_pseudo_mask == self.pseudo_token_id).nonzero().reshape((bz, self.num_prompt, 2))[:, :,
                          1].to(self.device)

        for bidx in range(bz):
            for i in range(self.num_prompt):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

        return raw_embeds

    def forward(self, input_ids, input_pseudo_mask):

        embeds = self.embed_input(input_ids, input_pseudo_mask)

        attention_mask = input_ids != self.pad_token_id

        output = self.model(inputs_embeds=embeds.to(self.device),
                            attention_mask=attention_mask.to(self.device))

        return output.logits


# ========================================= model end ===================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tuning_model = PTuningModel(device)
tuning_model.to(device)

# exponential lr
params = [{"params": tuning_model.prompt_encoder.parameters()}]
optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=0.0005)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)


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
        input_ids, seg_ids, label_ids, pseudo_mask = batch
        logits = tuning_model(input_ids, pseudo_mask)
        loss = compute_loss(logits, label_ids)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0 and step != 0:
            logger.info(f"epoch: {e} - step: {step} - loss: {loss.item()} - lr: {my_lr_scheduler.get_last_lr()[0]:.6f}")
        if step == len(train_generator) * 50:
            break
    my_lr_scheduler.step()

    tuning_model.eval()
    total, right = 0., 0.
    for batch in tqdm(test_generator):
        input_ids, _, label_ids, pseudo_mask = batch
        with torch.no_grad():
            logits = tuning_model(input_ids.to(device), pseudo_mask.to(device))

        logits = logits.cpu()
        pred = logits[:, mask_idx, [neg_id, pos_id]].argmax(dim=1)
        y_true = (label_ids[:, mask_idx] == pos_id).long()
        total += input_ids.size(0)
        right += (y_true == pred).sum().item()

    val_acc = right / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    logger.info(f"epoch: {e} - acc: {val_acc:.6f} - best_test_acc: {best_val_acc}")

