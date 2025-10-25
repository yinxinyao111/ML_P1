from model import built_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path
#from dataset import BilingualDataset, causal_mask

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
# Dataset related 
def get_ds(config):
    # retrieve raw English -> Russian dataset
    ds_raw = load_dataset(f"{config["datasource"]}", f"{config["lang_src"]}-{config["lang_tgt"]}", split = "train")
    
    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])
    
    # train-val split (with raw sentences)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # process raw sentences to tensors
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # find max sequence lengths
    max_len_src, max_len_tgt = 0, 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item["translation"][config["lang_src"]])
        tgt_ids = tokenizer_tgt.encode(item["translation"][config["lang_tgt"]])
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    print(f"max length of English sentence is {max_len_src}")
    print(f"max length of Russian sentence is {max_len_tgt}")
    
    # convert processed datasets to dataloaders
    train_dataloader = DataLoader(train_ds, batch_size = config["batch_size"], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = config["batch_size"], shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# Auxillary function for get_ds()
def get_or_build_tokenizer(config, ds, lang):
    # ex. tokenizer_ru.json
    tokenizer_path = Path(config["tokenizer_file"]).format(lang)
    if not Path.exists(tokenizer_path):
        # create tokenizer object
        tokenizer = Tokenizer(WordLevel(unk_token = "[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # create trainer object
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        # train the tokenizer with trainer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        # save tokenizer
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# Auxillary function for get_or_build_tokenizer()
def get_all_sentences(ds, lang):
    for item in ds:
        yield item["translation"]["lang"]

#-------------------------------------------------------------
# Model related
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Training on ", device)
    device = torch.device(device)
    
    # create weight folder
    Path(f"{config["datasource"]}_{config["model_folder"]}").mkdir(parents = True, exist_ok= True)
    
    # prepare dataloaders & tokenizers & model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    