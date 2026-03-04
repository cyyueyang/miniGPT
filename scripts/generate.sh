#!/bin/bash
# 推理脚本 - 使用前修改配置

# 指定 GPU
export CUDA_VISIBLE_DEVICES=0

# 模型路径
CHECKPOINT="./out/gpt/model.pt"

# 运行交互模式
python -m miniGPT.inference interactive \
    --checkpoint $CHECKPOINT \
    --device cuda

# 或者单条生成（取消下面注释，注释掉上面的交互命令）
# python -m miniGPT.inference generate \
#     --checkpoint $CHECKPOINT \
#     --prompt "Once upon a time, " \
#     --max-tokens 200 \
#     --device cuda
