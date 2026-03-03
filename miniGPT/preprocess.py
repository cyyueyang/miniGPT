#!/usr/bin/env python3
"""预处理大型文本文件为 token ID 文件"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from .bpe import get_encoder


def preprocess_text_file(input_path, output_path, chunk_size=10000000):
    """预处理文本文件为 token ID 数组
    
    Args:
        input_path: 输入文本文件路径
        output_path: 输出 .npy 文件路径
        chunk_size: 每次读取的字符数
    """
    encoder = get_encoder()
    eos_token = 50256
    
    if os.path.exists(output_path):
        print(f"输出文件已存在: {output_path}")
        response = input("是否覆盖? (y/n): ")
        if response.lower() != 'y':
            print("取消操作")
            return
    
    file_size = os.path.getsize(input_path)
    print(f"输入文件大小: {file_size / 1024**3:.2f} GB")
    print(f"输出文件: {output_path}")
    
    all_tokens = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        chunk_num = 0
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="处理中") as pbar:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                tokens = encoder.encode(chunk)
                tokens.append(eos_token)
                all_tokens.extend(tokens)
                
                pbar.update(len(chunk.encode('utf-8')))
                chunk_num += 1
                
                if chunk_num % 10 == 0:
                    print(f"已处理 {chunk_num} 个 chunk, 累计 token 数: {len(all_tokens):,}")
    
    token_array = np.array(all_tokens, dtype=np.uint16)
    np.save(output_path, token_array)
    
    print(f"预处理完成!")
    print(f"总 token 数: {len(all_tokens):,}")
    print(f"输出文件大小: {os.path.getsize(output_path) / 1024**3:.2f} GB")


def preprocess_dataset(input_dir, output_dir):
    """预处理整个数据集目录
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files_to_process = [
        ('train.txt', 'train_tokens.npy'),
        ('valid.txt', 'valid_tokens.npy'),
        ('test.txt', 'test_tokens.npy'),
    ]
    
    for input_name, output_name in files_to_process:
        input_path = os.path.join(input_dir, input_name)
        output_path = os.path.join(output_dir, output_name)
        
        if os.path.exists(input_path):
            print(f"\n处理文件: {input_name}")
            preprocess_text_file(input_path, output_path)
        else:
            print(f"\n跳过不存在的文件: {input_name}")


def main():
    parser = argparse.ArgumentParser(description="预处理文本文件为 token ID 数组")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 单个文件处理
    file_parser = subparsers.add_parser('file', help='处理单个文件')
    file_parser.add_argument('--input', required=True, help='输入文本文件路径')
    file_parser.add_argument('--output', required=True, help='输出 .npy 文件路径')
    file_parser.add_argument('--chunk-size', type=int, default=10000000, help='每次读取的字符数')
    
    # 数据集处理
    dataset_parser = subparsers.add_parser('dataset', help='处理整个数据集目录')
    dataset_parser.add_argument('--input-dir', required=True, help='输入目录路径')
    dataset_parser.add_argument('--output-dir', required=True, help='输出目录路径')
    
    args = parser.parse_args()
    
    if args.command == 'file':
        preprocess_text_file(args.input, args.output, args.chunk_size)
    elif args.command == 'dataset':
        preprocess_dataset(args.input_dir, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
