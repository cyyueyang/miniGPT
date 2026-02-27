import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math

class SwiGLU(nn.Module):
    def __init__(self, config):
        super(SwiGLU, self).__init__()
        hidden_dim = int(config.n_embd * 8 / 3)
        # 加速
        hidden_dim = ((hidden_dim + config.multiple_of - 1) // config.multiple_of) * config.multiple_of
        self.w_gate = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(self.w_up(x) * F.silu(self.w_gate(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.register_buffer("mask",
                             torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)
                                        .view(1, 1, config.block_size, config.block_size)))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        batchsize, seq_len, d_model = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q = q.view(batchsize, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2) # (bs, n_head, seq_len, n_dim)
        k = k.view(batchsize, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2) # (bs, n_head, seq_len, n_dim)
        v = v.view(batchsize, seq_len, self.n_head, d_model // self.n_head).transpose(1, 2) # (bs, n_head, seq_len, n_dim)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d_model // self.n_head))
        attn = attn.masked_fill(~self.mask[:, :, :seq_len, :seq_len], -1e9)
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(batchsize, seq_len, d_model)
        y = self.c_proj(y)

        return y

class RMSNorm(nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.n_embd))

    def forward(self, x):
        rms = torch.rsqrt(torch.pow(x, 2).mean(dim=-1, keepdim=True) + 1e-12)
        x_normed = x * rms * self.gamma
        return x_normed

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.norm1 = RMSNorm(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config)
        self.ffn = SwiGLU(config)

    def forward(self, x):
        _x = self.norm1(x)
        _x = self.attn(_x)
        x = _x + x
        _x = self.norm2(x)
        _x = self.ffn(_x)
        x = _x + x
        return x

