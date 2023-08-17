import torch
from torch import nn
from models.transformer.layer_norm import LayerNorm
from models.transformer.position_wise_ffn import PositionWiseFeedForward
from models.transformer.rotary_position_mha import MultiHeadAttention
from models.transformer.relative_position_mha import RPEMultiHeadAttention
from models.transformer.abs_pe import PositionalEncoding


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, norm_type, pe='absolute', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert norm_type in ('pre', 'post', 'rezero'), \
            "norm_type must be 'pre', 'post' or 'rezero'"
        assert pe in ('rotary', 'relative', 'absolute'), "rpe must be either 'rotary' or 'relative' or 'absolute'"
        if pe == 'rotary':
            self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        elif pe == 'relative':
            self.self_attention = RPEMultiHeadAttention(d_model=d_model, n_head=n_head)
        else:
            self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, pe=False)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head, pe=False)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.norm_type = norm_type  # post, pre, or rezero
        if self.norm_type == 'pre' or self.norm_type == 'post':
            self.norm1 = LayerNorm(d_model=d_model)
            self.norm2 = LayerNorm(d_model=d_model)
        else:
            self.res_weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, dec, enc, trg_mask, src_mask):
        if self.norm_type == 'post':
            x = self.norm1(self.dropout1(self.self_attention(query=dec, key=dec, value=dec, mask=trg_mask)) + dec)
            x = self.norm2(self.dropout2(self.enc_dec_attention(query=x, key=enc, value=enc, mask=src_mask)) + x)
            x = self.norm3(self.dropout3(self.ffn(x)) + x)
        elif self.norm_type == 'pre':
            x_pre_norm = self.norm1(dec)
            x = self.dropout1(self.self_attention(
                query=x_pre_norm, key=x_pre_norm, value=x_pre_norm, mask=trg_mask)) + dec
            x = self.dropout2(self.enc_dec_attention(query=self.norm2(x), key=enc, value=enc, mask=src_mask)) + x
            x = self.dropout3(self.ffn(self.norm3(x))) + x
        else:
            x = self.dropout1(self.self_attention(
                query=dec, key=dec, value=dec, mask=trg_mask) * self.res_weight) + dec
            x = self.dropout2(self.enc_dec_attention(
                query=x, key=enc, value=enc, mask=src_mask) * self.res_weight) + x
            x = self.dropout3(self.ffn(x) * self.res_weight) + x
        return x


class Decoder(nn.Module):

    def __init__(self, dec_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, norm_type='post', pe='absolute',
                 tie_emb=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tie_emb:
            self.register_parameter("emb", None)
        else:
            self.emb = nn.Embedding(dec_size, d_model, padding_idx=1)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, norm_type=norm_type, pe=pe)
                                     for _ in range(n_layers)])
        if pe == 'absolute':
            self.abs_pe = PositionalEncoding(d_model=d_model, dropout=drop_prob, max_len=2048)
        else:
            self.register_parameter("abs_pe", None)
        self.fc = nn.Linear(d_model, dec_size)

    def forward(self, trg, enc_hid, trg_mask, src_mask):
        if self.emb is not None:
            trg = self.emb(trg)
        if self.abs_pe is not None:
            trg = self.abs_pe(trg)
        else:
            trg = self.dropout(trg)
        for layer in self.layers:
            trg = layer(trg, enc_hid, trg_mask, src_mask)
        trg = self.fc(trg)
        return trg
