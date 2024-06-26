import torch
from torch import nn
from models.stable_transformer.layer_norm import LayerNorm
from models.stable_transformer.position_wise_ffn import PositionWiseFeedForward
from models.stable_transformer.mha import MultiHeadAttention
from models.stable_transformer.abs_pe import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    Single decoder layer in a transformer decoder.

    Performs multi-head self-attention, encoder-decoder attention, and
    position-wise feedforward operations. Supports layer normalization.

    Args:
        d_model (int): Embedding dimension size.
        ffn_hidden (int): Feedforward network hidden layer size.
        n_head (int): Number of attention heads.
        drop_prob (float): Dropout probability.
        norm_type (str): Type of layer normalization, 'pre' or 'post'.
        pe (str): Positional encoding type, 'rotary', 'relative' or 'absolute'.

    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, norm_type, norm_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert norm_type in ('pre', 'post', 'rezero'), \
            "norm_type must be 'pre', 'post' or 'rezero'"
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head,cross_attention=True)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.norm_type = norm_type  # post, pre, or rezero
        if self.norm_type == 'pre' or self.norm_type == 'post':
            self.norm1 = LayerNorm(d_model=d_model,bias=norm_bias)
            self.norm2 = LayerNorm(d_model=d_model,bias=norm_bias)
            self.norm3 = LayerNorm(d_model=d_model,bias=norm_bias)
        else:
            self.res_weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, dec, enc, trg_mask, src_mask):
        if self.norm_type == 'post':
            x = self.norm1(self.dropout1(self.self_attention(query=dec, mask=trg_mask)) + dec)
            x = self.norm2(self.dropout2(self.enc_dec_attention(query=x, key=enc, mask=src_mask)) + x)
            x = self.norm3(self.dropout3(self.ffn(x)) + x)
        elif self.norm_type == 'pre':
            x_pre_norm = self.norm1(dec)
            x = self.dropout1(self.self_attention(
                query=x_pre_norm, mask=trg_mask)) + dec
            x = self.dropout2(self.enc_dec_attention(query=self.norm2(x), key=enc, mask=src_mask)) + x
            x = self.dropout3(self.ffn(self.norm3(x))) + x
        else:
            x = self.dropout1(self.self_attention(
                query=dec, mask=trg_mask) * self.res_weight) + dec
            x = self.dropout2(self.enc_dec_attention(
                query=x, key=enc, mask=src_mask) * self.res_weight) + x
            x = self.dropout3(self.ffn(x) * self.res_weight) + x
        return x

class Decoder(nn.Module):
    """
    Transformer decoder module.

    Args:
        dec_size (int): Target vocabulary size.
        d_model (int): Embedding dimension size.
        ffn_hidden (int): Feedforward hidden layer size.
        n_head (int): Number of attention heads.
        n_layers (int): Number of decoder layers.
        drop_prob (float): Dropout probability.
        norm_type (str): Type of layer normalization.
        pe (str): Positional encoding type.
        tie_emb (bool): Tie input embedding matrix as decoder embedding.

    """
    def __init__(self, dec_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, norm_type='post',
                 tie_emb=False,norm_bias=True,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if tie_emb:
            self.register_parameter("emb", None)
        else:
            self.emb = nn.Embedding(dec_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, norm_type=norm_type, norm_bias=norm_bias)
                                     for _ in range(n_layers)])
        self.abs_pe = PositionalEncoding(d_model=d_model, drop_prob=drop_prob, max_len=2048)
        if norm_type == 'pre':
            self.final_norm = LayerNorm(d_model, bias=norm_bias)
        else:
            self.register_parameter("final_norm", None)
        self.fc = nn.Linear(d_model, dec_size)

    def forward(self, trg, enc_hid, trg_mask, src_mask):
        if self.emb is not None:
            trg = self.emb(trg)
        trg = self.abs_pe(trg)
        for layer in self.layers:
            trg = layer(trg, enc_hid, trg_mask, src_mask)
        if self.final_norm is not None:
            trg = self.final_norm(trg)
        trg = self.fc(trg)
        return trg