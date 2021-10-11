import logging
import os
from pathlib import Path
import json
import numpy as np
try:
    import pandas as pd
except:
    print("not install pandas")
from tqdm import tqdm
from collections import defaultdict


ROOT_PATH = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
workspace = Path(ROOT_PATH) / ".."

# data
data_dir = Path(ROOT_PATH) / "data"
model_dir = Path(ROOT_PATH) / "model"

bert_data_path = workspace / 'db__pytorch_pretrained_bert'
common_data_path = workspace / 'db__common_dataset'

# bert
bert_vocab_path = bert_data_path / 'bert-base-chinese' / 'vocab.txt'
bert_model_path = bert_data_path / 'bert-base-chinese'

# roberta
roberta_wwm_path = bert_data_path / "chinese_wwm_ext_L-12_H-768_A-12"

roberta_large_model_path = bert_data_path / 'chinese_Roberta_bert_wwm_large_ext_pytorch'
bert_wwm_pt_path = bert_data_path / "chinese_wwm_ext_pytorch"
roberta_wwm_pt_path = bert_data_path / "chinese_roberta_wwm_ext_pytorch"

# open dataset
open_dataset_path = common_data_path / "open_dataset"


###############################################
# log
###############################################

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f'begin progress ...')


