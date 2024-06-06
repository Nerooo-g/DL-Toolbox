import torch
from torch import nn
from models.stable_transformer.layer_norm import LayerNorm
from models.stable_transformer.position_wise_ffn import PositionWiseFeedForward
from models.stable_transformer.mha import MultiHeadAttention
from models.stable_transformer.abs_pe import PositionalEncoding

def make_mask(seq, cross_attention=False, is_causal=False, trg_seq=None, padding_value=0, dtype="float"):
    """
    Create a mask tensor for padding and attention mechanisms.

    Args:
        dtype: data type of the mask tensor
        seq (torch.Tensor): Input sequence tensor of shape [batch_size, length].
        cross_attention (bool): If True, return mask tensor for cross-attention.
        is_causal (bool): If True, return mask tensor for causal attention.
        trg_seq (torch.Tensor): Target sequence tensor of shape [batch_size, length] for cross-attention.
        padding_value (int): Value used for padding in the sequences.

    Returns:
        torch.Tensor: Mask tensor of shape [batch_size, src_length, trg_length].
    """
    if not cross_attention:
        output_tensor = torch.ones((seq.shape[0], seq.shape[-1], seq.shape[-1]), dtype=torch.bool)
        for i in range(seq.shape[0]):
            # Find positions in the row that are equal to the padding value
            zero_indices = (seq[i] == padding_value).nonzero(as_tuple=True)[0]
            # Set corresponding rows and columns to False for each zero position
            for idx in zero_indices:
                output_tensor[i, idx, :] = False
                output_tensor[i, :, idx] = False
        if is_causal:
            # Generate a causal mask
            causal_mask = torch.tril(torch.ones((seq.shape[-1], seq.shape[-1]), dtype=torch.bool))
            # Apply the causal mask to the output tensor
            output_tensor = output_tensor & causal_mask
    else:
        output_tensor = torch.ones((seq.shape[0], seq.shape[-1], trg_seq.shape[-1]), dtype=torch.bool)

        for i in range(seq.shape[0]):
            # Find positions in the first matrix (seq) that are equal to the padding value
            zero_indices_A = (seq[i] == padding_value).nonzero(as_tuple=True)[0]
            # Find positions in the second matrix (trg_seq) that are equal to the padding value
            zero_indices_B = (trg_seq[i] == padding_value).nonzero(as_tuple=True)[0]
            # Set corresponding positions to False for each zero position in seq
            for idx in zero_indices_A:
                output_tensor[i, idx, :] = False
            # Set corresponding positions to False for each zero position in trg_seq
            for idx in zero_indices_B:
                output_tensor[i, :, idx] = False
    if dtype == "float":
        output_tensor = (1.0 - output_tensor.float()) * torch.finfo(torch.float32).min
    return output_tensor

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

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, norm_type,norm_bias=True, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert norm_type in ('pre', 'post', 'rezero'), \
            "norm_type must be 'pre', 'post' or 'rezero'"
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm_type = norm_type  # post, pre, or rezero
        if self.norm_type == 'pre' or self.norm_type == 'post':
            self.norm1 = LayerNorm(d_model=d_model, bias=norm_bias)
            self.norm2 = LayerNorm(d_model=d_model, bias=norm_bias)
        else:
            self.res_weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x, src_mask):
        if self.norm_type == 'post':
            x = self.norm1(self.dropout1(self.attention(query=x, mask=src_mask)) + x)
            x = self.norm2(self.dropout2(self.ffn(x)) + x)
        elif self.norm_type == 'pre':
            x_pre_norm = self.norm1(x)
            x = self.dropout1(self.attention(query=x_pre_norm, key=x_pre_norm, mask=src_mask)) + x
            x = self.dropout2(self.ffn(self.norm2(x))) + x
        else:
            x = self.dropout1(self.attention(query=x, key=x, mask=src_mask) * self.res_weight) + x
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

    def __init__(self, enc_size, d_model, ffn_hidden, n_head, n_layers, drop_prob, norm_type='post',
                 tie_emb=False, norm_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if tie_emb:
            self.register_parameter("emb", None)
        else:
            self.emb = nn.Embedding(enc_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob, norm_type=norm_type, norm_bias=norm_bias)
                                     for _ in range(n_layers)])
        self.abs_pe = PositionalEncoding(d_model=d_model, drop_prob=drop_prob, max_len=2048)
        if norm_type == 'pre':
            self.final_norm = LayerNorm(d_model, bias=norm_bias)
        else:
            self.register_parameter("final_norm", None)

    def forward(self, x, src_mask):
        if self.emb is not None:
            x = self.emb(x)
        x = self.abs_pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x