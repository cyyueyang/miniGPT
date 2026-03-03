#!/usr/bin/env python3
"""GPT 模型训练脚本"""

import os
import sys
import json
import argparse
import math
import torch

from .dataset import GPTDataset, MemMappedGPTDataset
from .model import GPT
from .utils import set_seed, setup_logging, CfgNode, dict_to_cfgnode
from .trainer import Trainer
from .bpe import get_encoder


def get_default_config():
    """获取默认配置"""
    C = CfgNode()

    C.system = CfgNode()
    C.system.seed = 3407
    C.system.work_dir = './out/gpt'

    C.data = CfgNode()
    C.data.block_size = 128
    C.data.train_file = 'data/TinyStoriesV2-GPT4-train.txt'
    C.data.val_file = 'data/TinyStoriesV2-GPT4-valid.txt'

    C.model = GPT.get_default_config()

    C.trainer = Trainer.get_default_config()

    return C


def load_config(config_path=None):
    """加载配置，支持从 JSON 文件加载"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = dict_to_cfgnode(config_dict)
    else:
        config = get_default_config()
    return config


def save_checkpoint(model, config, checkpoint_path, iteration=None):
    """保存模型检查点"""
    checkpoint = {
        'model': model.state_dict(),
        'config': config.to_dict(),
        'iteration': iteration,
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, device='cpu'):
    """加载模型检查点"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    print("检查点加载完成")
    return checkpoint.get('iteration', 0)


def create_dataset(config, tokenizer):
    """创建训练集和验证集"""
    train_file = config.data.train_file
    val_file = config.data.val_file if hasattr(config.data, 'val_file') else None
    
    print(f"训练文件: {train_file}")
    
    # 训练集
    if train_file.endswith('.npy'):
        train_dataset = MemMappedGPTDataset(config.data, train_file)
    else:
        with open(train_file, 'r', encoding='utf-8') as f:
            train_text = f.read()
        train_dataset = GPTDataset(config.data, train_text, tokenizer)
    
    # 验证集
    val_dataset = None
    if val_file and os.path.exists(val_file):
        print(f"验证文件: {val_file}")
        if val_file.endswith('.npy'):
            val_dataset = MemMappedGPTDataset(config.data, val_file)
        else:
            with open(val_file, 'r', encoding='utf-8') as f:
                val_text = f.read()
            val_dataset = GPTDataset(config.data, val_text, tokenizer)
    
    return train_dataset, val_dataset


def validate_model(model, dataset, device='auto', batch_size=16):
    """在验证集上评估模型"""
    if dataset is None:
        return None
    
    from torch.utils.data import DataLoader
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    model.train()
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'tokens': total_tokens,
    }


def train(config):
    """主训练函数"""
    print("配置:")
    print(config)
    
    setup_logging(config)
    set_seed(config.system.seed)
    
    # 加载 tokenizer
    tokenizer = get_encoder()
    
    # 创建数据集
    train_dataset, val_dataset = create_dataset(config, tokenizer)
    
    # 更新模型配置
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    
    # 创建模型
    model = GPT(config.model)
    
    # 检查是否恢复训练
    checkpoint_dir = config.system.work_dir
    checkpoint_path = os.path.join(checkpoint_dir, 'model.pt')
    start_iteration = 0
    
    if os.path.exists(checkpoint_path):
        response = input(f"找到检查点 {checkpoint_path}，是否恢复训练? (y/n): ")
        if response.lower() == 'y':
            start_iteration = load_checkpoint(checkpoint_path, model, config.trainer.device)
            print(f"从迭代 {start_iteration} 恢复训练")
    
    # 创建训练器
    trainer = Trainer(config.trainer, model, train_dataset)
    trainer.iter_num = start_iteration
    
    # 训练回调
    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        
        if trainer.iter_num % 500 == 0 and trainer.iter_num > 0:
            model.eval()
            with torch.no_grad():
                prompt = "Once upon a time, "
                input_ids = tokenizer.encode(prompt)
                x = torch.tensor([input_ids], dtype=torch.long, device=trainer.device)
                
                y = model.generate(
                    x, 
                    max_new_tokens=100, 
                    temperature=1.0, 
                    do_sample=True, 
                    top_k=40
                )
                
                generated_text = tokenizer.decode(y[0].tolist())
                print(f"\n{'=' * 40}")
                print(f"样本 at step {trainer.iter_num}:")
                print(f"提示: {prompt}")
                print(f"生成: {generated_text[len(prompt):]}")
                print(f"{'=' * 40}\n")
            
            if val_dataset is not None:
                val_results = validate_model(model, val_dataset, trainer.device)
                if val_results:
                    print(f"验证集 - 损失: {val_results['loss']:.4f}, 困惑度: {val_results['perplexity']:.2f}")
            
            save_checkpoint(model, config, checkpoint_path, trainer.iter_num)
            model.train()
    
    trainer.set_callback("on_batch_end", batch_end_callback)
    
    print(f"开始训练...")
    print(f"工作目录: {config.system.work_dir}")
    print(f"设备: {trainer.device}")
    
    trainer.run()
    
    # 保存最终模型
    save_checkpoint(model, config, checkpoint_path, trainer.iter_num)
    print("训练完成!")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="GPT 模型训练")
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--work-dir', help='工作目录')
    parser.add_argument('--max-iters', type=int, help='最大训练迭代次数')
    parser.add_argument('--batch-size', type=int, help='批处理大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--device', help='设备 (auto, cpu, cuda)')
    parser.add_argument('--resume', action='store_true', help='自动恢复训练')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.work_dir:
        config.system.work_dir = args.work_dir
    if args.max_iters:
        config.trainer.max_iters = args.max_iters
    if args.batch_size:
        config.trainer.batch_size = args.batch_size
    if args.lr:
        config.trainer.lr = args.lr
    if args.device:
        config.trainer.device = args.device
    
    train(config)


if __name__ == '__main__':
    main()
