from tokenizer import Tokenizer
from models import TransformerDecoderLightning
from dataloader import TextDataset, Collator
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
import glob
import os
import sentencepiece as spm
import re

book_list = glob.glob("/Users/rachel/PycharmProjects/Transformer/books/*.txt")
ckpt = '/Users/rachel/PycharmProjects/lightning_logs/version_8/checkpoints/epoch=11-step=20160.ckpt'
# ckpt = None

def load_text_file(filename):
    lines = []
    with open(filename, 'r') as fin:
        for line in fin:
            if line != '':
                lines.append(line.strip())
    lines = ' '.join(lines)
    lines = lines.split('. ')
    lines = [a for a in lines if len(a) > 0]
    lines = [li + '.' for li in lines]
    new_lines = [lines[0]]
    for line in lines[1:]:
        if len(new_lines[-1]) < 64:
            new_lines[-1] += f' {line}'
        else:
            new_lines.append(line)
    return new_lines


def load_books(book_list):
    lines = []
    for book in book_list:
        lines += load_text_file(book)
    return lines


lines = load_books(book_list)

# prepend = 'Today is my first day. My name is'
# lines = [
#         f'{prepend} George',
#         f'{prepend} Jason',
#         f'{prepend} Liz',
#         f'{prepend} Panda',
#         f'{prepend} Tiger',
#         f'{prepend} Buffalo',
#         ]*512
sp = spm.SentencePieceProcessor()
sp.Load('spms/spm_1024.model')



# Create the dataset
dataset = TextDataset(text=lines,
                      tokenizer=sp,
                      max_seq_len=128)

collate_fn = Collator(in_seq_pad_id=sp.piece_to_id('<pad>'),
                      out_seq_pad_id=-1)
train_loader = DataLoader(dataset,
                          collate_fn=collate_fn,
                          batch_size=32, shuffle=True,
                          )

# Create the model
model = TransformerDecoderLightning(num_tokens=len(sp),
                                    per_head_dim=256,
                                    num_heads=2,
                                    num_layers=2,
                                    ignore_index=-1)
# model.to(mps_device)

trainer = pl.Trainer(
                    # limit_train_batches=100,
                     max_epochs=10000,
                     accelerator="cpu",
                     default_root_dir="/Users/rachel/PycharmProjects/llm_checkpoints/"
                     )
trainer.fit(model=model,
            train_dataloaders=train_loader,
            ckpt_path=ckpt)
