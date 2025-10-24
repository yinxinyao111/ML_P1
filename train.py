from model import built_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask

import torch
import torch.nn as nn
import torchtext.datasets as datasets
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
# -----------------------------------------------------------------

def get_ds(config):
    # retrieve raw English -> Russian dataset
    ds_raw = load_dataset(f"{config["datasource"]}", f"{config["lang_src"]}-{config["lang_tgt"]}", split = "train")
    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    # train-val split
    train_ds_size = int(0.9 * len(ds_raw))
    