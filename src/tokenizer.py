from typing import Optional
import torch


class Tokenizer:
    def __init__(self,
                 tokenizer_file: Optional[str] = None,
                 text: Optional[list] = None,
                 special_tokens: Optional[list] = None):
        if tokenizer_file is not None:
            self.load(tokenizer_file)
        elif text is not None:
            self.tokens = self.create_tokenizer(text)
            self.special_tokens = special_tokens
            if self.special_tokens is None:
                self.tokens.extend(self.create_specials())

            # create the lookup tables
            self.token_to_id = self.create_token_map()
            self.id_to_token = {value: key for key, value in self.token_to_id.items()}
        else:
            assert False, "Either tokenizer_file or text to create tokens from must be given"

    def create_token_map(self):
        token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
        return token_to_id

    def create_specials(self):
        special_tokens = ["<sos>",
                          "<eos>",
                          "<unk>",
                          "<pad>"]
        return special_tokens

    def create_tokenizer(self, text: list):
        # text_samples is a list of text samples
        chars = set()
        for sample in text:
            tokens = set([x for x in sample])
            chars = tokens.union(chars)
        # add space token
        return list(chars)

    @property
    def sos_id(self):
        return self.token_to_id["<sos>"]

    @property
    def eos_id(self):
        return self.token_to_id["<eos>"]

    @property
    def pad_id(self):
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self):
        return self.token_to_id["<unk>"]

    @property
    def num_tokens(self):
        return len(self.token_to_id.keys())

    def encode(self, text: str):
        encoded = []
        for char in text:
            if char not in self.token_to_id:
                print(f"{char} not found in tokens {self.token_to_id}")
                char = "<unk>"
            encoded.append(self.token_to_id[char])
        return encoded

    def decode(self, ids: list):
        # ids is a list of integers
        tokens = "".join(self.id_to_token[idx] for idx in ids)
        return tokens

    def save(self, out_filename):
        # save the tokenizer to given filename
        save_dict = {'token_to_id': self.token_to_id,
                     'id_to_token': self.id_to_token}
        torch.save(save_dict, out_filename)

    def load(self, in_filename):
        # load the tokenizer to given filename
        save_dict = torch.load(in_filename)
        self.token_to_id = save_dict['token_to_id']
        self.id_to_token = save_dict['id_to_token']
