import os
import json
import regex as re
import requests


def bytes_to_unicode():
    """返回字节到 unicode 字符的映射表"""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """返回单词中所有相邻字符对"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    """BPE 编码器"""
    
    def __init__(self, encoder, bpe_merges):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        """对单个 token 应用 BPE 合并"""
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda x: self.bpe_ranks.get(x, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            new_word = tuple(new_word)
            word = new_word

            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """将文本编码为 token ids 列表"""
        bpe_idx = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            bpe_idx.extend([self.encoder[bpe_token] for bpe_token in token_merged])
        return bpe_idx

    def decode(self, bpe_idx):
        """将 token ids 解码为文本"""
        tokens_merged = [self.decoder[token] for token in bpe_idx if token in self.decoder]
        tokens_flat = ''.join(tokens_merged)
        token_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat if c in self.byte_decoder])
        text = token_bytes.decode("utf-8", errors='replace')
        return text


def get_file(local_file, remote_file):
    """下载文件到本地缓存"""
    if not os.path.isfile(local_file):
        print(f"下载 {remote_file} 到 {local_file}")
        response = requests.get(remote_file)
        with open(local_file, 'wb') as f:
            f.write(response.content)


def get_encoder():
    """获取 GPT-2 BPE 编码器"""
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.minigpt')
    os.makedirs(cache_dir, exist_ok=True)

    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)

    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r') as f:
        bpe_data = f.read()

    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1] if merge_str]
    encoder = Encoder(encoder, bpe_merges)
    return encoder


class BPETokenizer:
    """BPE Tokenizer 包装类，提供统一接口"""
    
    def __init__(self):
        self.encoder = get_encoder()

    def encode(self, text):
        """编码文本为 token ids 列表"""
        return self.encoder.encode(text)

    def decode(self, ids):
        """解码 token ids 为文本
        
        Args:
            ids: token id 列表或 tensor
        """
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self.encoder.decode(ids)

    def __call__(self, text, return_tensors="pt"):
        """支持类似 HuggingFace 的调用方式"""
        assert return_tensors == "pt"
        assert isinstance(text, str)
        idx = self.encoder.encode(text)
        import torch
        return torch.tensor([idx], dtype=torch.long)
