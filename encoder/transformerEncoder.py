import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from Bio import PDB

class SelfAttention(nn.Cell):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Dense(embed_dim, embed_dim)
        self.W_k = nn.Dense(embed_dim, embed_dim)
        self.W_v = nn.Dense(embed_dim, embed_dim)
        self.W_o = nn.Dense(embed_dim, embed_dim)

        self.softmax = nn.Softmax(axis=-1)
        self.scale = ms.Tensor(self.head_dim ** -0.5, dtype=ms.float32)

        # self.W_q = torch_nn.Linear(embed_dim, embed_dim)
        # self.W_k = torch_nn.Linear(embed_dim, embed_dim)
        # self.W_v = torch_nn.Linear(embed_dim, embed_dim)
        # self.W_o = torch_nn.Linear(embed_dim, embed_dim)

    def construct(self, x):
        batch_size, seq_len, embed_dim = x.shape
        q = self.W_q(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2)
        k = self.W_k(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2)
        v = self.W_v(x).reshape((batch_size, seq_len, self.num_heads, self.head_dim)).swapaxes(1, 2)

        attn_scores = ops.BatchMatMul()(q, k.swapaxes(-1, -2)) * self.scale
        attn_weights = self.softmax(attn_scores)
        attn_output = ops.BatchMatMul()(attn_weights, v)

        attn_output = attn_output.swapaxes(1, 2).reshape((batch_size, seq_len, embed_dim))
        return self.W_o(attn_output)

        # attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_output = torch.matmul(attn_weights, v)
        # return self.W_o(attn_output.reshape(batch_size, seq_len, embed_dim))

class TransformerEncoder(nn.Cell):
    def __init__(self, embed_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm([embed_dim])
        self.norm2 = nn.LayerNorm([embed_dim])
        self.feed_forward = nn.SequentialCell([
            nn.Dense(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dense(embed_dim * 4, embed_dim)
        ])

        # self.feed_forward = torch_nn.Sequential(
        #     torch_nn.Linear(embed_dim, embed_dim * 4),
        #     torch_nn.ReLU(),
        #     torch_nn.Linear(embed_dim * 4, embed_dim)
        # )

    def construct(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x

        # attn_out = self.attention(x)
        # x = torch_nn.LayerNorm(embed_dim)(x + attn_out)
        # ff_out = self.feed_forward(x)
        # return torch_nn.LayerNorm(embed_dim)(x + ff_out)
