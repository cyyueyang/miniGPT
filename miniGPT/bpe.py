import torch
import os
import json
import regex as re


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


