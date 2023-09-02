import torch
import lightning.pytorch as pl
from torch import optim
import numpy as np
import sentencepiece as spm

class SelfAttention(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 causal: bool = True):
        super().__init__()
        self.pkey = torch.nn.Linear(in_dim, hidden_dim)
        self.pquery = torch.nn.Linear(in_dim, hidden_dim)
        self.pvalue = torch.nn.Linear(in_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=2)
        self.causal = causal

    def forward(self, x, padding_mask=None):
        # x is of shape (B, T, D=in_dim)
        # Batch, Time, Dimension
        # padding mask is of shape (B, T)
        # key is of shape (B, T, D=hidden_dim)
        key = self.pkey(x)

        # query is of shape(B, T, D=hidden_dim)
        query = self.pquery(x)

        # get the attention matrix A = Q * (K^t)
        # key is (B, D, T)
        key = torch.transpose(key, 1, 2)

        # attention is (B, T, T)
        attention = torch.bmm(query, key)

        # import pdb; pdb.set_trace()
        # mask padding tokens
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(2)
            attention = attention.masked_fill_(padding_mask, -1e8)

        # mask future tokens
        if self.causal:
            mask = torch.triu(torch.ones_like(attention), diagonal=1)
            mask = mask.to(torch.bool)
            attention = attention.masked_fill_(mask, -1e8)

        # Perform softmax along dim=1
        scale= torch.sqrt(torch.tensor(x.shape[-1]))
        attention = self.softmax(attention/scale)

        # value is (B, T, D=hidden_dim)
        value = self.pvalue(x)

        # perform attention
        out = torch.bmm(attention, value)

        # output is of (B, T, D)
        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 per_head_dim: int,
                 num_heads: int,
                 causal: bool = True,
                 ):
        super().__init__()
        self.blocks = torch.nn.ModuleList([SelfAttention(in_dim=in_dim, hidden_dim=per_head_dim) \
                                           for _ in range(num_heads)])
        self.causal = causal

    def forward(self, x, input_mask=None):
        # x is of shape (B, T, D=in_dim)
        y = [block(x, input_mask) for block in self.blocks]
        # y is of shape (B, T, D=per_head_dim*num_heads)
        y = torch.cat(y, dim=-1)
        return y


class FeedForward(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, in_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.blocks(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 per_head_dim: int,
                 num_heads: int):
        super().__init__()
        dims = per_head_dim*num_heads
        self.self_attention = MultiHeadAttention(in_dim=dims,
                                                 per_head_dim=per_head_dim,
                                                 num_heads=num_heads)
        # self.self_attention = torch.nn.MultiheadAttention(embed_dim=per_head_dim*num_heads,
        #                                                   num_heads=num_heads,
        #                                                   dropout=0.1,
        #                                                   batch_first=True,)
        self.norm1 = torch.nn.LayerNorm(dims)
        self.ff = FeedForward(in_dim=dims, hidden_dim=2*dims)
        self.norm2 = torch.nn.LayerNorm(dims)

    def forward(self, x, input_mask=None):
        y = x + self.self_attention(x, input_mask=input_mask)
        # causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        # causal_mask = causal_mask < 0
        # y, _ = self.self_attention(x, x, x,
        #                             key_padding_mask=input_mask,
        #                             is_causal=True,
        #                             attn_mask=causal_mask,
        #                             need_weights=False)
        # y = y + x
        y = self.norm1(y)
        y = self.ff(y) + y
        y = self.norm2(y)
        return y


class PositionalEncoding(torch.nn.Module):
    def __init__(self,
                 out_dim: int,
                 max_length: int=256,
                 ):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_length, out_dim)

    def forward(self, positions):
        # positions is of shape (B, max_seq_len), integer
        return self.embedding(positions)


class TransformerDecoder(torch.nn.Module):
    def __init__(self,
                 per_head_dim: int,
                 num_heads: int,
                 num_layers: int,
                 num_tokens: int,
                 max_input_len: int = 256,
                 ):
        super().__init__()
        total_dim = per_head_dim * num_heads
        self.pos_embedding = PositionalEncoding(out_dim=total_dim,
                                                max_length=max_input_len)
        self.transformers = torch.nn.ModuleList([TransformerBlock(per_head_dim=per_head_dim,
                                                                  num_heads=num_heads) for _ in range(num_layers)])
        self.embedding = torch.nn.Embedding(num_tokens, total_dim)
        self.linear = torch.nn.Linear(total_dim, num_tokens)


    def forward(self, x, input_mask=None):
        # x is of (B, T), where each element is an integer [0, num_tokens-1]
        # get the token embeddings
        y = self.embedding(x)

        # get positional embeddings
        p_embed = self.pos_embedding(torch.arange(x.shape[1]))

        # add positional embeddings
        y = y + p_embed

        # go through transformer blocks
        for block in self.transformers:
            y = block(y, input_mask=input_mask)

        # Final linear layer
        y = self.linear(y)

        return y


# define the LightningModule
class TransformerDecoderLightning(pl.LightningModule):
    def __init__(self,
                 num_tokens: int,
                 per_head_dim: int = 128,
                 num_heads: int = 1,
                 num_layers: int = 4,
                 ignore_index: int = -1,
                 max_input_len: int = 256,
                 tokenizer: str = './spms/spm_1024.model'
                 ):
        super().__init__()
        self.model = TransformerDecoder(per_head_dim=per_head_dim,
                                        num_heads=num_heads,
                                        num_layers=num_layers,
                                        num_tokens=num_tokens,
                                        max_input_len=max_input_len)
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer)
        eos_id = self.tokenizer.piece_to_id('<eos>')
        weight = torch.ones(num_tokens)
        weight[eos_id] = 0.1
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index,
                                                 weight=weight)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['inputs']
        y = batch['labels']
        input_pad_mask = batch['input_pad_mask']
        x_hat = self.model(x, input_pad_mask)
        # x_hat is (B, T, C). Need to reshape to (B, C, T)
        x_hat = x_hat.transpose(1, 2)
        # import pdb; pdb.set_trace()
        # y is (B, T)
        loss = self.ce_loss(x_hat, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def infer_step(self, x, top_k=1):
        # Take one step of inference
        # x is of shape (1, T)
        # out_step is of shape (1, T, C)
        out_step = self.model(x)
        # get the most probable last token
        # next_token = torch.argmax(out_step, dim=-1)

        # top-K sampling
        probs, next_token = torch.topk(out_step, k=top_k, dim=-1)
        next_token = next_token[0, -1, :]
        next_token = next_token.numpy()
        # next_token = np.random.choice(next_token)
        idx = torch.multinomial(probs[0, -1, :], 1)
        next_token = next_token[idx]
        return next_token

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=3e-5)
        return optimizer

