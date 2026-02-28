import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import CfgNode

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

class GPT(nn.Module):
    """
    GPT model
    """
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.block_size = None
        C.vocab_size = None
        C.multiple_of = 256
        return C

    def __init__(self, config: CfgNode):
        super(GPT, self).__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given   # 要么给模型类型 要么给参数

        if type_given:
            config.merge_from_dict({
                # GPT-1
                'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768, multiple_of=256),  # 117M params
                # GPT-2 configs
                'gpt2': dict(n_layer=12, n_head=12, n_embd=768, multiple_of=256),  # 124M params   和 上面配置相同 参数量不同的原因是因为 词表不同
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024, multiple_of=256),  # 350M params
                'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280, multiple_of=256),  # 774M params
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600, multiple_of=256),  # 1558M params
                # Gophers
                'gopher-44m': dict(n_layer=8, n_head=16, n_embd=512, multiple_of=256),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192, multiple_of=256),
                'gpt-micro': dict(n_layer=4, n_head=4, n_embd=128, multiple_of=256),
                'gpt-nano': dict(n_layer=3, n_head=3, n_embd=48, multiple_of=256),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm_f = RMSNorm(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, param in self.transformer.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(param, mean=0, std=0.02 / math.sqrt(config.n_layer))

        n_parms = sum(param.numel() for param in self.parameters())
        n_parms_without_lm_head = sum(param.numel() for param in self.transformer.parameters())
        print(f"n_parms M: {n_parms / 1e6}")
        print(f"n_parms_without_lm_head: {n_parms_without_lm_head / 1e6}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, RMSNorm):
            torch.nn.init.ones_(m.gamma)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        # bais 也不需要正则化
        whitelist_weight_modules = (torch.nn.Linear,)
        # RMSNorm 权重: 用于缩放归一化后的输出，加正则化会破坏归一化效果
        # Embedding 权重：词汇嵌入通常是稀疏使用的，L2 会惩罚罕见词的学习
        blacklist_weight_modules = (RMSNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
               full_param_name = f"{mn}.{pn}" if mn else pn
               if pn.endswith('bias'):
                   no_decay.add(full_param_name)
               elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                   decay.add(full_param_name)
               elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                   no_decay.add(full_param_name)
               elif pn.endswith('gamma'):
                   no_decay.add(full_param_name)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, "inter_parms must be 0"
        assert len(param_dict.keys() - union_params) == 0, "union_params must equal to params"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.lr, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        bs, seq_len = idx.shape
        assert seq_len <= self.block_size
        pos = torch.arange(start=0, end=seq_len, device=device, dtype=torch.long).unsqueeze(0)
        token_emb = self.transformer.wte(idx)
        position_emb = self.transformer.wpe(pos)
        x = token_emb + position_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False, eos_token_id=50256):
        bs = idx.shape[0]
        finished = torch.zeros(bs, device=idx.device, dtype=torch.bool)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size]
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k)
                logits[logits < v[:, [-1]]] = -1e9

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1) # 贪心搜索最大的
            idx_next = torch.where(finished.unsqueeze(-1), eos_token_id, idx_next)
            idx = torch.cat((idx_cond, idx_next), dim=1)
            finished = finished | (idx_next.squeeze(-1) == eos_token_id)
            if torch.all(finished):
                break
        return idx














