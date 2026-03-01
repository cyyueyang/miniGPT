import os
import torch
from torch.utils.data import Dataset
from .utils import CfgNode


class GPTDataset(Dataset):

    def __init__(self, config, data, encoder):
        self.config = config
        self.encoder = encoder

        print(f"Encoding {len(data)} chars")
        self.tokens = encoder.encode(data)
        self.token_count = len(self.tokens)
        print(f"Token count: {self.token_count}")
        self.eos_token = 50256 # <|endoftext|>

    def get_vocab_size(self):
        return 50257

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        # 每个样本 需要 bolcksize + 1 个token  x y 错位
        return self.token_count - self.config.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.config.block_size + 1]
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        return x, y






