#!/usr/bin/env python3
"""GPT 模型推理脚本"""

import os
import sys
import argparse
import torch
from .model import GPT
from .bpe import BPETokenizer, get_encoder
from .utils import CfgNode, dict_to_cfgnode


def load_model(checkpoint_path, config=None, device='auto'):
    """从检查点加载模型"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"加载模型从: {checkpoint_path}")
    print(f"设备: {device}")
    
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
    print(f"参数数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, model_config


def generate_text(
    model,
    prompt,
    tokenizer,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    do_sample=True,
    device='auto'
):
    """生成文本"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # 编码提示
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # 生成
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
            eos_token_id=50256
        )
    
    # 解码
    generated_text = tokenizer.decode(y[0].tolist())
    
    return generated_text, input_ids, y[0].tolist()


def batch_generate(
    model,
    prompts,
    tokenizer,
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    do_sample=True,
    batch_size=4,
    device='auto'
):
    """批量生成文本"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_input_ids = []
        
        for prompt in batch_prompts:
            input_ids = tokenizer.encode(prompt)
            batch_input_ids.append(input_ids)
        
        # 找到最大长度并填充
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_ids = []
        for ids in batch_input_ids:
            if len(ids) < max_len:
                padded = ids + [50256] * (max_len - len(ids))
            else:
                padded = ids
            padded_ids.append(padded)
        
        x = torch.tensor(padded_ids, dtype=torch.long, device=device)
        
        with torch.no_grad():
            y = model.generate(
                x,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                eos_token_id=50256
            )
        
        for j, (prompt, output_ids) in enumerate(zip(batch_prompts, y.tolist())):
            original_len = len(batch_input_ids[j])
            generated_ids = output_ids[original_len:]
            
            generated_text = tokenizer.decode(generated_ids)
            full_text = tokenizer.decode(output_ids)
            
            results.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'full_text': full_text,
                'input_ids': batch_input_ids[j],
                'output_ids': output_ids
            })
        
        print(f"处理进度: {min(i+batch_size, len(prompts))}/{len(prompts)}")
    
    return results


def interactive_mode(model, tokenizer, device='auto'):
    """交互式模式"""
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print("\n" + "="*50)
    print("GPT 交互式模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    print("="*50)
    
    while True:
        try:
            prompt = input("\n请输入提示 (或输入 'help'): ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            elif prompt.lower() in ['help', 'h', '?']:
                print("\n可用命令:")
                print("  quit/exit/q - 退出")
                print("  help/h/? - 显示此帮助")
                print("\n生成参数设置:")
                print("  使用格式: 提示文本 [max_tokens=100] [temp=1.0] [top_k=40] [sample=true]")
                continue
            elif prompt == '':
                continue
            
            # 解析参数
            parts = prompt.split()
            text_parts = []
            max_new_tokens = 100
            temperature = 1.0
            top_k = 40
            do_sample = True
            
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    key = key.lower()
                    if key == 'max_tokens':
                        max_new_tokens = int(value)
                    elif key == 'temp':
                        temperature = float(value)
                    elif key == 'top_k':
                        top_k = None if value.lower() == 'none' else int(value)
                    elif key == 'sample':
                        do_sample = value.lower() in ['true', 'yes', 'y', '1']
                else:
                    text_parts.append(part)
            
            prompt_text = ' '.join(text_parts)
            
            if not prompt_text:
                print("错误: 请输入提示文本")
                continue
            
            print(f"\n生成参数: max_tokens={max_new_tokens}, temp={temperature}, top_k={top_k}, sample={do_sample}")
            print("生成中...")
            
            generated_text, input_ids, output_ids = generate_text(
                model=model,
                prompt=prompt_text,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=do_sample,
                device=device
            )
            
            print(f"\n{'='*50}")
            print(f"提示: {prompt_text}")
            print(f"生成: {generated_text[len(prompt_text):]}")
            print(f"{'='*50}")
            
        except KeyboardInterrupt:
            print("\n\n中断，退出交互模式")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="GPT 模型推理")
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 单次生成
    generate_parser = subparsers.add_parser('generate', help='单次文本生成')
    generate_parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    generate_parser.add_argument('--prompt', required=True, help='提示文本')
    generate_parser.add_argument('--max-tokens', type=int, default=100, help='最大生成 token 数')
    generate_parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    generate_parser.add_argument('--top-k', type=int, help='top-k 采样参数')
    generate_parser.add_argument('--no-sample', action='store_true', help='使用贪婪解码')
    generate_parser.add_argument('--device', default='auto', help='设备')
    
    # 批量生成
    batch_parser = subparsers.add_parser('batch', help='批量文本生成')
    batch_parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    batch_parser.add_argument('--prompts-file', required=True, help='提示文本文件路径')
    batch_parser.add_argument('--output-file', required=True, help='输出文件路径')
    batch_parser.add_argument('--max-tokens', type=int, default=100, help='最大生成 token 数')
    batch_parser.add_argument('--temperature', type=float, default=1.0, help='温度参数')
    batch_parser.add_argument('--top-k', type=int, help='top-k 采样参数')
    batch_parser.add_argument('--no-sample', action='store_true', help='使用贪婪解码')
    batch_parser.add_argument('--batch-size', type=int, default=4, help='批处理大小')
    batch_parser.add_argument('--device', default='auto', help='设备')
    
    # 交互模式
    interactive_parser = subparsers.add_parser('interactive', help='交互式模式')
    interactive_parser.add_argument('--checkpoint', required=True, help='模型检查点路径')
    interactive_parser.add_argument('--device', default='auto', help='设备')
    
    args = parser.parse_args()
    
    # 初始化 tokenizer
    tokenizer = BPETokenizer()
    
    if args.command == 'generate':
        model, _ = load_model(args.checkpoint, device=args.device)
        
        generated_text, input_ids, output_ids = generate_text(
            model=model,
            prompt=args.prompt,
            tokenizer=tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.no_sample,
            device=args.device
        )
        
        print("\n" + "="*60)
        print(f"提示: {args.prompt}")
        print(f"生成: {generated_text[len(args.prompt):]}")
        print(f"完整文本: {generated_text}")
        print(f"输入 token 数: {len(input_ids)}")
        print(f"输出 token 数: {len(output_ids)}")
        print("="*60)
        
    elif args.command == 'batch':
        model, _ = load_model(args.checkpoint, device=args.device)
        
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        print(f"读取 {len(prompts)} 个提示")
        
        results = batch_generate(
            model=model,
            prompts=prompts,
            tokenizer=tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=not args.no_sample,
            batch_size=args.batch_size,
            device=args.device
        )
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for i, result in enumerate(results):
                f.write(f"=== 示例 {i+1} ===\n")
                f.write(f"提示: {result['prompt']}\n")
                f.write(f"生成: {result['generated_text']}\n")
                f.write(f"完整文本: {result['full_text']}\n\n")
        
        print(f"结果已保存到: {args.output_file}")
        
    elif args.command == 'interactive':
        model, _ = load_model(args.checkpoint, device=args.device)
        interactive_mode(model, tokenizer, device=args.device)
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
