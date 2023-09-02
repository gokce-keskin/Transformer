from tokenizer import Tokenizer
from models import TransformerDecoderLightning
from dataloader import TextDataset, Collator
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
import os
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.Load('spms/spm_1024.model')
ckpt = '/Users/rachel/PycharmProjects/lightning_logs/version_8/checkpoints/epoch=11-step=20160.ckpt'


# # Create the model
# model = TransformerDecoderLightning(num_tokens=my_tokenizer.num_tokens,
#                                     ignore_index=-1)

model = TransformerDecoderLightning.load_from_checkpoint(ckpt,
                                                         num_tokens=len(sp),
                                                         per_head_dim=256,
                                                         num_heads=2,
                                                         num_layers=2,
                                                         )
model.eval()
prompt = sp.encode('When was the last time ')
prompt = [sp.piece_to_id('<sos>')] + prompt
prompt = torch.tensor(prompt).to(torch.long).unsqueeze(0)
for i in range(200):
    next_token = model.infer_step(prompt, top_k=3)
    prompt = torch.cat( [prompt,
                         torch.tensor([next_token]).to(torch.long).unsqueeze(0)],
                        dim=-1)
    text = [sp.decode(prompt[0].numpy().tolist())]
    if next_token == sp.piece_to_id('<eos>'):
        break
print(text)
