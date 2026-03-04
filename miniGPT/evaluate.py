import os
import sys
import argparse
import math
import json
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import GPT
from .bpe import BPETokenizer
from .dataset import GPTDataset, MemMappedGPTDataset
from .utils import CfgNode, dict_to_cfgnode


def calculate_perplexity(model, dataloader, device='auto'):
    """计算模型在数据集上的困惑度"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算困惑度"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            _, loss = model(x, y)
            
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    return perplexity, avg_loss, total_tokens


def calculate_accuracy(model, dataloader, device='auto'):
    """计算模型在数据集上的准确率"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算准确率"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            logits, _ = model(x, y)
            predictions = torch.argmax(logits, dim=-1)
            
            correct = (predictions == y).sum().item()
            tokens = y.numel()
            
            total_correct += correct
            total_tokens += tokens
    
    if total_tokens > 0:
        accuracy = total_correct / total_tokens
    else:
        accuracy = 0.0
    
    return accuracy, total_correct, total_tokens


def evaluate_model(
    model,
    dataset,
    batch_size=16,
    num_workers=0,
    device='auto'
):
    """全面评估模型"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批处理大小: {batch_size}")
    
    print("\n计算困惑度...")
    perplexity, avg_loss, total_tokens = calculate_perplexity(
        model, dataloader, device=device
    )
    
    print("\n计算准确率...")
    accuracy, total_correct, _ = calculate_accuracy(
        model, dataloader, device=device
    )
    
    bpc = avg_loss / math.log(2) if avg_loss < float('inf') else float('inf')
    
    metrics = {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'bpc': bpc,
        'total_tokens': total_tokens,
        'total_correct': total_correct,
        'dataset_size': len(dataset),
        'vocab_size': dataset.get_vocab_size() if hasattr(dataset, 'get_vocab_size') else 50257,
        'block_size': dataset.get_block_size() if hasattr(dataset, 'get_block_size') else None,
    }
    
    return metrics


def load_model_and_dataset(checkpoint_path, eval_data_path, config=None, device='auto'):
    """加载模型和数据集用于评估"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"加载模型从: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        if isinstance(model_config, dict):
            model_config = dict_to_cfgnode(model_config)
    elif config is not None:
        model_config = config
    else:
        raise ValueError("需要提供模型配置，或者检查点必须包含配置")
    
    model = GPT(model_config)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("模型加载完成")
    
    # 加载 tokenizer
    tokenizer = BPETokenizer()
    
    # 创建数据集
    print(f"创建评估数据集从: {eval_data_path}")
    
    if eval_data_path.endswith('.npy'):
        eval_config = CfgNode()
        eval_config.block_size = model_config.block_size
        dataset = MemMappedGPTDataset(eval_config, eval_data_path)
    else:
        eval_config = CfgNode()
        eval_config.block_size = model_config.block_size
        
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        dataset = GPTDataset(eval_config, text, tokenizer.encoder)
    
    print(f"评估数据集大小: {len(dataset)}")
    
    return model, dataset, tokenizer


def compare_models(model_paths, eval_data_path, configs=None, device='auto'):
    """比较多个模型的性能"""
    results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"\n{'='*60}")
        print(f"评估模型 {i+1}/{len(model_paths)}: {model_path}")
        print(f"{'='*60}")
        
        try:
            config = configs[i] if configs and i < len(configs) else None
            
            model, dataset, tokenizer = load_model_and_dataset(
                model_path, eval_data_path, config, device
            )
            
            metrics = evaluate_model(
                model,
                dataset,
                batch_size=16,
                num_workers=0,
                device=device
            )
            
            results.append({
                'model_path': model_path,
                'metrics': metrics,
            })
            
            print(f"\n模型 {model_path} 评估结果:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"评估模型 {model_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GPT 模型评估")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 单个模型评估
    eval_parser = subparsers.add_parser('evaluate', help='评估单个模型')
    eval_parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    eval_parser.add_argument('--eval-data', required=True, help='评估数据路径')
    eval_parser.add_argument('--batch-size', type=int, default=16, help='批处理大小')
    eval_parser.add_argument('--num-workers', type=int, default=0, help='数据加载器工作线程数')
    eval_parser.add_argument('--device', default='auto', help='设备')
    eval_parser.add_argument('--output', help='输出结果文件路径')
    
    # 模型比较
    compare_parser = subparsers.add_parser('compare', help='比较多个模型')
    compare_parser.add_argument('--checkpoints', required=True, nargs='+', help='模型检查点路径列表')
    compare_parser.add_argument('--eval-data', required=True, help='评估数据路径')
    compare_parser.add_argument('--batch-size', type=int, default=16, help='批处理大小')
    compare_parser.add_argument('--num-workers', type=int, default=0, help='数据加载器工作线程数')
    compare_parser.add_argument('--device', default='auto', help='设备')
    compare_parser.add_argument('--output', required=True, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        model, dataset, tokenizer = load_model_and_dataset(
            args.checkpoint, args.eval_data, device=args.device
        )
        
        metrics = evaluate_model(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
        
        print("\n" + "="*60)
        print("评估结果:")
        print("="*60)
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        if args.output:
            result = {
                'checkpoint': args.checkpoint,
                'eval_data': args.eval_data,
                'metrics': metrics,
                'timestamp': str(datetime.datetime.now()),
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n结果已保存到: {args.output}")
    
    elif args.command == 'compare':
        results = compare_models(
            args.checkpoints,
            args.eval_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
        
        comparison = {
            'models': results,
            'eval_data': args.eval_data,
            'timestamp': str(datetime.datetime.now()),
            'summary': {}
        }
        
        summary = {}
        for i, result in enumerate(results):
            model_name = os.path.basename(result['model_path'])
            summary[model_name] = {
                'perplexity': result['metrics']['perplexity'],
                'accuracy': result['metrics']['accuracy'],
                'avg_loss': result['metrics']['avg_loss'],
                'bpc': result['metrics']['bpc'],
            }
        
        comparison['summary'] = summary
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"\n比较结果已保存到: {args.output}")
        
        print("\n" + "="*60)
        print("模型比较摘要:")
        print("="*60)
        for model_name, metrics in summary.items():
            print(f"\n模型: {model_name}")
            print(f"  困惑度: {metrics['perplexity']:.4f}")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  平均损失: {metrics['avg_loss']:.4f}")
            print(f"  BPC: {metrics['bpc']:.4f}")
        
        if results:
            best_by_perplexity = min(results, key=lambda x: x['metrics']['perplexity'])
            best_by_accuracy = max(results, key=lambda x: x['metrics']['accuracy'])
            
            print("\n" + "="*60)
            print("最佳模型:")
            print("="*60)
            print(f"最低困惑度: {os.path.basename(best_by_perplexity['model_path'])} "
                  f"({best_by_perplexity['metrics']['perplexity']:.4f})")
            print(f"最高准确率: {os.path.basename(best_by_accuracy['model_path'])} "
                  f"({best_by_accuracy['metrics']['accuracy']:.4f})")
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
