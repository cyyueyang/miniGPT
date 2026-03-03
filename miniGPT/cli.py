#!/usr/bin/env python3
"""miniGPT 命令行接口"""

import os
import sys
import json
import argparse
from .utils import CfgNode, dict_to_cfgnode


def load_config(config_path):
    """从 JSON 文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return dict_to_cfgnode(config_dict)


def train_command(args):
    """训练命令"""
    from .train import get_default_config, train
    
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
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


def inference_command(args):
    """推理命令"""
    from .inference import main as inference_main
    
    inference_args = ['generate']
    
    if args.prompt:
        inference_args.extend(['--prompt', args.prompt])
    if args.checkpoint:
        inference_args.extend(['--checkpoint', args.checkpoint])
    if args.max_tokens:
        inference_args.extend(['--max-tokens', str(args.max_tokens)])
    if args.temperature:
        inference_args.extend(['--temperature', str(args.temperature)])
    if args.top_k:
        inference_args.extend(['--top-k', str(args.top_k)])
    if args.no_sample:
        inference_args.append('--no-sample')
    if args.device:
        inference_args.extend(['--device', args.device])
    if args.interactive:
        inference_args[0] = 'interactive'
    
    sys.argv = [sys.argv[0]] + inference_args
    inference_main()


def evaluate_command(args):
    """评估命令"""
    from .evaluate import main as evaluate_main
    
    evaluate_args = ['evaluate']
    
    if args.checkpoint:
        evaluate_args.extend(['--checkpoint', args.checkpoint])
    if args.eval_data:
        evaluate_args.extend(['--eval-data', args.eval_data])
    if args.batch_size:
        evaluate_args.extend(['--batch-size', str(args.batch_size)])
    if args.num_workers:
        evaluate_args.extend(['--num-workers', str(args.num_workers)])
    if args.device:
        evaluate_args.extend(['--device', args.device])
    if args.output:
        evaluate_args.extend(['--output', args.output])
    
    sys.argv = [sys.argv[0]] + evaluate_args
    evaluate_main()


def preprocess_command(args):
    """预处理命令"""
    from .preprocess import main as preprocess_main
    
    preprocess_args = []
    
    if args.input and args.output:
        preprocess_args.extend(['file', '--input', args.input, '--output', args.output])
    if args.input_dir and args.output_dir:
        preprocess_args.extend(['dataset', '--input-dir', args.input_dir, '--output-dir', args.output_dir])
    if args.chunk_size:
        preprocess_args.extend(['--chunk-size', str(args.chunk_size)])
    
    if preprocess_args:
        sys.argv = [sys.argv[0]] + preprocess_args
        preprocess_main()
    else:
        print("错误: 需要指定输入和输出路径")
        sys.exit(1)


def config_command(args):
    """配置命令"""
    if args.list:
        config_dir = 'configs'
        if os.path.exists(config_dir):
            print("可用的配置文件:")
            for file in os.listdir(config_dir):
                if file.endswith('.json'):
                    print(f"  - {file}")
        else:
            print(f"配置目录不存在: {config_dir}")
    
    elif args.show and args.name:
        config_path = os.path.join('configs', args.name)
        if not config_path.endswith('.json'):
            config_path += '.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = json.load(f)
            print(f"配置文件: {args.name}")
            print(json.dumps(config_content, indent=2, ensure_ascii=False))
        else:
            print(f"配置文件不存在: {config_path}")
    
    elif args.create and args.name:
        from .model import GPT
        from .trainer import Trainer
        
        model_config = GPT.get_default_config()
        trainer_config = Trainer.get_default_config()
        
        config = {
            "system": {
                "seed": 3407,
                "work_dir": f"./out/{args.name}"
            },
            "data": {
                "block_size": 128,
                "train_file": "data/TinyStoriesV2-GPT4-train.txt",
                "val_file": "data/TinyStoriesV2-GPT4-valid.txt"
            },
            "model": model_config.to_dict() if hasattr(model_config, 'to_dict') else model_config.__dict__,
            "trainer": trainer_config.__dict__ if hasattr(trainer_config, '__dict__') else trainer_config
        }
        
        config_path = os.path.join('configs', args.name)
        if not config_path.endswith('.json'):
            config_path += '.json'
        
        os.makedirs('configs', exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"配置文件已创建: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="miniGPT 命令行接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s train --config configs/gpt_mini.json
  %(prog)s inference --prompt "Hello world" --checkpoint out/gpt/model.pt
  %(prog)s evaluate --checkpoint out/gpt/model.pt --eval-data data/valid.txt
  %(prog)s preprocess --input data/train.txt --output data/train_tokens.npy
  %(prog)s config --list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', help='配置文件路径')
    train_parser.add_argument('--work-dir', help='工作目录')
    train_parser.add_argument('--max-iters', type=int, help='最大训练迭代次数')
    train_parser.add_argument('--batch-size', type=int, help='批处理大小')
    train_parser.add_argument('--lr', type=float, help='学习率')
    train_parser.add_argument('--device', help='设备')
    
    # 推理命令
    inference_parser = subparsers.add_parser('inference', help='模型推理')
    inference_parser.add_argument('--prompt', help='提示文本')
    inference_parser.add_argument('--checkpoint', help='模型检查点路径')
    inference_parser.add_argument('--max-tokens', type=int, default=100, help='最大生成 token 数')
    inference_parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    inference_parser.add_argument('--top-k', type=int, help='top-k 采样参数')
    inference_parser.add_argument('--no-sample', action='store_true', help='使用贪婪解码')
    inference_parser.add_argument('--device', default='auto', help='设备')
    inference_parser.add_argument('--interactive', action='store_true', help='交互模式')
    
    # 评估命令
    evaluate_parser = subparsers.add_parser('evaluate', help='模型评估')
    evaluate_parser.add_argument('--checkpoint', help='模型检查点路径')
    evaluate_parser.add_argument('--eval-data', help='评估数据路径')
    evaluate_parser.add_argument('--batch-size', type=int, default=16, help='批处理大小')
    evaluate_parser.add_argument('--num-workers', type=int, default=0, help='工作线程数')
    evaluate_parser.add_argument('--device', default='auto', help='设备')
    evaluate_parser.add_argument('--output', help='输出文件路径')
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='数据预处理')
    preprocess_parser.add_argument('--input', help='输入文件路径')
    preprocess_parser.add_argument('--output', help='输出文件路径')
    preprocess_parser.add_argument('--input-dir', help='输入目录路径')
    preprocess_parser.add_argument('--output-dir', help='输出目录路径')
    preprocess_parser.add_argument('--chunk-size', type=int, default=10000000, help='块大小')
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_group = config_parser.add_mutually_exclusive_group()
    config_group.add_argument('--list', action='store_true', help='列出所有配置文件')
    config_group.add_argument('--show', action='store_true', help='显示配置文件内容')
    config_group.add_argument('--create', action='store_true', help='创建新配置文件')
    config_parser.add_argument('--name', help='配置文件名')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    elif args.command == 'preprocess':
        preprocess_command(args)
    elif args.command == 'config':
        config_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
