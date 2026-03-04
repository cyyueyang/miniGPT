#!/bin/bash
# 训练脚本 - 使用前修改配置

# 指定 GPU (0 是第一张卡，1 是第二张，以此类推)
export CUDA_VISIBLE_DEVICES=0

# 训练命令
python -m miniGPT.train \
    --config configs/gpt_mini.json \
    --work-dir ./out/gpt \
    --max-iters 10000 \
    --batch-size 64 \
    --device cuda
