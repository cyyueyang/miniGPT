#!/bin/bash
# 数据预处理脚本

python -m miniGPT.preprocess file \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/train_tokens.npy
