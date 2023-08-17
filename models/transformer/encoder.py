import torch
from torch import nn
from models.transformer.layer_norm import LayerNorm
from models.transformer.position_wise_ffn import PositionWiseFeedForward
from models.transformer.rotary_position_mha import MultiHeadAttention
from models.transformer.relative_position_mha import RPEMultiHeadAttention
from models.transformer.abs_pe import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    Single encoder layer in a transformer encoder.

    Performs multi-head attention and position-wise feedforward operations.
    Supports layer normalization before or after these operations.

    Args:
        d_model (int): Embedding dimension size.
        ffn_hidden (int): Feedforward network hidden layer size.
        n_head (int): Number of attention heads.
        drop_prob (float): Dropout probability.
        norm_type (str): Type of layer normalization, 'pre' or 'post'.
        pe (str): Positional encoding type, 'rotary', 'relative' or 'absolute'.

    """
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, norm_type, pe='absolute', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert norm_type in ('pre', 'post', 'rezero'), \
            "norm_type must be 'pre', 'post' or 'rezero'"
        assert pe in ('rotary', 'relative', 'absolute'), "rpe must be either 'rotary' or 'relative' or 'absolute'"
        if pe == 'rotary':
            self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        elif pe == 'relative':
            self.attention = RPEMultiHeadAttention(d_model=d_model, n_head=n_head)
        else:
            self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, pe=False)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm_type = norm_type  # post, pre, or rezero
        if self.norm_type == 'pre' or self.norm_type == 'post':
            self.norm1 = LayerNorm(d_model=d_model)
            self.norm2 = LayerNorm(d_model=d_model)
        else:
            self.res_weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x, src_mask):
        if self.norm_type == 'post':
            x = self.norm1(self.dropout1(self.attention(query=x, key=x, value=x, mask=src_mask)) + x)
            x = self.norm2(self.dropout2(self.ffn(x)) + x)
        elif self.norm_type == 'pre':
            x_pre_norm = self.norm1(x)
            x = self.dropout1(self.attention(query=x_pre_norm, key=x_pre_norm, value=x_pre_norm, mask=src_mask)) + x
            x = self.dropout2(self.ffn(self.norm2(x))) + x
        else:
            x = self.dropout1(self.attention(query=x, key=x, value=x, mask=src_mask) * self.res_weight) + x
            x = self.dropout2(self.ffn(x) * self.res_weight) + x
        return x


class Encoder(nn.Module):
    """
    Transformer encoder module.

    Args:
        enc_size (int): Source vocabulary size.
        d_model (int): Embedding dimension size.
        ffn_hidden (int): Feedforward hidden layer size.
        n_head (int): Number of attention heads.
        n_layers (int): Number of encoder layers.
        drop_prob (float): Dropout probability.
        norm_type (str): Type of layer normalization.
        pe (str): Positional encoding type.
        tie_emb (bool): Tie input embedding matrix as decoder embedding.

    """
    def __init__(self, enc_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, norm_type='post', pe='absolute',
                 tie_emb=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tie_emb:
            self.register_parameter("emb", None)
        else:
            self.emb = nn.Embedding(enc_size, d_model, padding_idx=1)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, norm_type=norm_type, pe=pe)
                                     for _ in range(n_layers)])
        if pe == 'absolute':
            self.abs_pe = PositionalEncoding(d_model=d_model, drop_prob=drop_prob, max_len=2048)
        else:
            self.register_parameter("abs_pe", None)

    def forward(self, x, src_mask):
        if self.emb is not None:
            x = self.emb(x)
        if self.abs_pe is not None:
            x = self.abs_pe(x)
        else:
            x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
