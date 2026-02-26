from secrets import token_bytes

import torch
import os
import json
import regex as re
import requests

def bytes_to_unicode():
    """
    让字节看起来更舒服
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(c) for c in cs]
    d = dict(zip(bs, cs))

    return d

def get_pairs(word):
    """
    返回单词被切分为pairs, eg. (w, o, r, d) -> {(w, o), (o, r), (r, d)}
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder():
    def __init__(self, encoder, bpe_merges):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder   # token字符串-> token id
        self.decoder = {v: k for k, v in self.encoder.items()} # token id -> token 字符串

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        # token 只有单个字符
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda x: self.bpe_ranks.get(x, float("inf")))
            if bigram not in self.bpe_ranks:
                break # 没有bigram可以再被合并了

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
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
        bpe_idx = []

        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            bpe_idx.extend([self.encoder[bpe_token] for bpe_token in token_merged])

        return bpe_idx

    def encode_and_show_word(self, text):
        bpe_idx = []
        parts = []
        tokens = re.findall(self.pat, text)
        for token in tokens:
            token_bytes = token.encode("utf-8")
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            bpe_idx.extend([self.encoder[bpe_token] for bpe_token in token_merged])
            parts.append({
                'token': token,
                'token_bytes': token_bytes,
                'token_translated': token_translated,
                'token_merged': token_merged,
                'token_ix': [self.encoder[bpe_token] for bpe_token in token_merged]
            })
        out = {
            'bpe_idx': bpe_idx,
            'parts': parts,
            'tokens': tokens,
        }
        return out

    def decode(self, bpe_idx):
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = ''.join(tokens_merged)
        token_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        text = token_bytes.decode("utf-8", errors='replace')
        return text

def get_file(local_file, remote_file):
    if not os.path.isfile(local_file):
        print(f"downloading {remote_file} to {local_file}")
        response = requests.get(remote_file)
        open(local_file, 'wb').write(response.content)

def get_encoder():
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.minigpt')
    os.makedirs(cache_dir, exist_ok=True)

    encoder_local_file = os.path.join(cache_dir, 'encoder.json')
    encoder_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
    get_file(encoder_local_file, encoder_remote_file)
    with open(encoder_local_file, 'r') as f:
        encoder = json.load(f)
    assert len(encoder) == (50000 + 256 + 1)

    vocab_local_file = os.path.join(cache_dir, 'vocab.bpe')
    vocab_remote_file = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'
    get_file(vocab_local_file, vocab_remote_file)
    with open(vocab_local_file, 'r') as f:
        bpe_data = f.read()

    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    assert len(bpe_merges) == 50000

    encoder = Encoder(encoder, bpe_merges)
    return encoder

