import torch
from torch import nn
from torch.cuda.amp import autocast

from models.transformer.decoder import Decoder
from models.transformer.encoder import Encoder


class Transformer(nn.Module):
    """
    Transformer model from Attention is All You Need paper.

    Args:
        enc_size (int): Source class size.
        dec_size (int): Target class size.
        d_model (int): Embedding dimension size.
        ffn_hidden (int): Feedforward hidden layer size.
        n_head (int): Number of attention heads.
        n_layers (int): Number of encoder/decoder layers.
        drop_prob (float): Dropout probability.
        norm_type (str): Type of layer normalization.
        pe (str): Positional encoding type.
        tie_emb (bool): Tie encoder/decoder embeddings.
        pad_idx (int): Padding index.

    """
    def __init__(self, enc_size, dec_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, norm_type='post',
                 pe='absolute', tie_emb=False, pad_idx=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_idx = pad_idx
        if tie_emb:
            assert enc_size == dec_size, "when tie embeddings, the encoder's class size must equal decoder's"
            self.emb = nn.Embedding(enc_size, d_model, padding_idx=1)
        else:
            self.register_parameter("emb", None)
        self.encoder = Encoder(enc_size=enc_size, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers, norm_type=norm_type, pe=pe, tie_emb=tie_emb)

        self.decoder = Decoder(dec_size=dec_size, d_model=d_model, n_head=n_head, ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers, norm_type=norm_type, pe=pe, tie_emb=tie_emb)

    @autocast()
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        if self.emb is not None:
            src = self.emb(src)
            trg = self.emb(trg)
        enc_hid = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_hid, trg_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).cuda()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
