import torch
import os
import torch.nn as nn
import torch.nn.functional as F

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

