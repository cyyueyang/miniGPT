import os
import numpy as np
import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """用于小规模文本的 GPT 数据集，一次性加载所有数据到内存"""

    def __init__(self, config, data, tokenizer):
        """
        Args:
            config: 配置对象，需包含 block_size
            data: 文本字符串或文件路径
            tokenizer: 编码器，需有 encode 方法
        """
        self.config = config
        self.tokenizer = tokenizer
        self.eos_token = 50256
        
        # 判断 data 是文件路径还是文本字符串
        if isinstance(data, str) and os.path.isfile(data):
            with open(data, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = data
            
        # 编码为 tokens
        self.tokens = self.tokenizer.encode(text)
        self.tokens.append(self.eos_token)
        self.token_count = len(self.tokens)
        
        # 转换为 tensor
        self.tokens_tensor = torch.tensor(self.tokens, dtype=torch.long)

    def get_vocab_size(self):
        return 50257

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        # 每个样本需要 block_size + 1 个 token (x 和 y 错位一位)
        return max(0, self.token_count - self.config.block_size)

    def __getitem__(self, idx):
        chunk = self.tokens_tensor[idx:idx + self.config.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class MemMappedGPTDataset(Dataset):
    """用于大规模文本的 GPT 数据集，使用内存映射读取预处理的 token 文件"""

    def __init__(self, config, token_file):
        """
        Args:
            config: 配置对象，需包含 block_size
            token_file: 预处理的 .npy 文件路径
        """
        self.config = config
        self.eos_token = 50256
        
        # 加载内存映射的 numpy 数组
        self.tokens = np.load(token_file, mmap_mode='r')
        self.token_count = len(self.tokens)
        
        print(f"加载 token 文件: {token_file}")
        print(f"Token 数量: {self.token_count:,}")

    def get_vocab_size(self):
        return 50257

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return max(0, self.token_count - self.config.block_size)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.config.block_size + 1]
        chunk_tensor = torch.tensor(chunk.copy(), dtype=torch.long)
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        return x, y
