import torch
import torch.nn.functional as F
from torch import nn


def concat(tensor):
    """
    inverse function of self.split(tensor : torch.Tensor)

    :param tensor: [batch_size, head, length, d_tensor]
    :return: [batch_size, length, d_model]
    """
    batch_size, head, length, d_tensor = tensor.size()
    d_model = head * d_tensor

    tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
    return tensor


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head,qkv_same_dim=True,cross_attention=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        if qkv_same_dim and not cross_attention:
            self.w_proj = nn.Linear(d_model, d_model * 3)
            self.register_parameter("w_q", None)
            self.register_parameter("w_k", None)
            self.register_parameter("w_v", None)
        elif cross_attention:
            self.w_q = nn.Linear(d_model, d_model)
            self.w_kv= nn.Linear(d_model, d_model*2)
            self.register_parameter("w_k", None)
            self.register_parameter("w_v", None)
        else:
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        # assert d_model % n_head == 0

    def forward(self, query, key=None, value=None, mask=None):
        # 1. dot product with weight matrices
        if hasattr(self, "w_proj"):
            qkv = self.w_proj(query)
            query, key, value = torch.chunk(qkv,3, dim=-1)
        elif hasattr(self, "w_kv"):
            query, kv = self.w_q(query), self.w_kv(key)
            key, value = torch.chunk(kv,2, dim=-1)
        else:
            query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)

        # 2. split tensor by number of heads
        query, key, value = self.split(query), self.split(key), self.split(value)

        # 3. do scale dot product to compute similarity
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,dropout_p=0.1)

        # 4. concat and pass to linear layer
        out = concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

