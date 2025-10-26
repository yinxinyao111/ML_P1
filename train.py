from model import build_transformer
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

# data visualization
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
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
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# Auxillary function for get_ds()
def get_or_build_tokenizer(config, ds, lang):
    # ex. tokenizer_ru.json
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
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
        yield item["translation"][lang]

#-------------------------------------------------------------
# training related
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples = 2):
    model.eval()
    count = 0
    
    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80
    
    source_texts = []
    expected = []
    predicted = []
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len)
            
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()
    
# Auxillary func for run_validation()
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")
    
    # precompute encoder output to reuse it for every decoder time step
    encoder_output = model.encode(source, source_mask)
    # decoder initialization with [SOS]
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    # generate tokens
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build decoder mask
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calc output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get next token
        prob = model.project(out[:, -1]) # (batch = 1, vocab_size)
        _, next_word = torch.max(prob, dim = 1) # returns (max_val, index)
        # add next_word to the new round of decoder_input
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)
        ], dim = 1) # (batch, seq_len + 1)
        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0) # (seq_len)
#-------------------------------------------------------------
# Model related
def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # device setup
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Training on ", device)
    device = torch.device(device)
    
    # create weight folder
    Path(f"{config["datasource"]}_{config["model_folder"]}").mkdir(parents = True, exist_ok= True)
    
    # prepare dataloaders & tokenizers & model
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tensorboard
    writer = SummaryWriter(config["experiment_name"])
    
    # Optimizer & loss fn
    optimizer = torch.optim.Adam(model.parameters(), lr = config["lr"], eps = 1e-9)
    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer_src.token_to_id("[PAD]"), label_smoothing = 0.1).to(device)
    
    # training setup
    initial_epoch, global_step = 0, 0
    
    preload = config["preload"]
    model_filename = latest_weights_file_path(config) if preload == "latest" else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f"Loading model{model_filename}")
        # load all states
        state = torch.load(model_filename)
        # from all states load model, epoch, optimizer, global step states
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    # training
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"Processing epoch {epoch}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch['encoder_mask'].to(device) 
            decoder_mask = batch['decoder_mask'].to(device) 

            # pass through model
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) # (batch, seq_len, vocab_size)
            
            # retrieve label
            label = batch["label"].to(device) # (batch, seq_len)
            
            # compute loss
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # display loss, rounded to 3 deci precision, padded to 6 length
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # log the loss
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            global_step += 1
        
        # validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config["seq_len"], device, lambda msg: batch_iterator.write(msg), global_step, writer)
        
        # save model per epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

# File execution
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)