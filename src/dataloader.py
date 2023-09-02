import torch
import random
from torch.utils.data import Dataset
from tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self,
                 text: list,
                 tokenizer: Tokenizer,
                 max_seq_len: int = 256,
                 ):
        super().__init__()
        self.text = text
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sos_id = self.tokenizer.piece_to_id("<sos>")
        self.eos_id = self.tokenizer.piece_to_id("<eos>")
        self.pad_id = self.tokenizer.piece_to_id("<pad>")


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(self.text[idx])[:self.max_seq_len-1]
        if len(encoded) > self.max_seq_len-1:
            start_idx = random.randint(0,
                                       len(encoded)-self.max_seq_len-1)
            encoded = encoded[start_idx:start_idx+self.max_seq_len-2]
        in_seq = [self.sos_id] + encoded
        out_seq = encoded + [self.eos_id]

        data = {
                'in_seq': in_seq,
                'out_seq': out_seq
                }
        return data


class Collator():
    def __init__(self,
                 in_seq_pad_id: int = -1,
                 out_seq_pad_id: int = -1):
        self.in_seq_pad_id = in_seq_pad_id
        self.out_seq_pad_id = out_seq_pad_id

    def __call__(self, data: list):
        # data is a list of dicts, each dict having in_seq and out_seq keys
        in_seqs = [d['in_seq'] for d in data]
        out_seqs = [d['out_seq'] for d in data]
        max_in_seq_len = max([len(d) for d in in_seqs])
        max_out_seq_len = max([len(d) for d in out_seqs])

        batch_size = len(in_seqs)
        input_pad_mask = torch.ones((batch_size, max_in_seq_len)).to(torch.bool)

        inputs = self.in_seq_pad_id * torch.ones((batch_size, max_in_seq_len))
        labels = self.out_seq_pad_id * torch.ones((batch_size, max_out_seq_len))
        # pad the in_seqs with in_seq_pad_id
        for idx in range(batch_size):
            in_seq = in_seqs[idx]
            out_seq = out_seqs[idx]
            inputs[idx, :len(in_seq)] = torch.tensor(in_seq)
            labels[idx, :len(out_seq)] = torch.tensor(out_seq)
            input_pad_mask[idx, :len(in_seq)] = False

        out_data = {'inputs': inputs.to(torch.long),
                    'labels': labels.to(torch.long),
                    'input_pad_mask': input_pad_mask}

        return out_data


