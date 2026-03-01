import os
import sys

import torch

from miniGPT.dataset import GPTDataset
from miniGPT.model import GPT
from miniGPT.utils import *
from miniGPT.trainer import *
from miniGPT.bpe import *

def get_config() -> CfgNode:
    C = CfgNode()

    C.system = CfgNode()
    C.system.seed = 3407
    C.system.word_dir = './out/gpt'

    C.data = CfgNode()
    C.data.block_size = 128

    C.model = GPT.get_default_config()

    C.trainer = Trainer.get_default_config()

    return C

if __name__ == '__main__':
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    text = open('data/TinyStoriesV2-GPT4-train.txt', 'r', encoding='utf-8').read()
    encoder = get_encoder()
    train_dataset = GPTDataset(config, text, encoder)
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    tokenizer = BPETokenizer
    trainer = Trainer(config.trainer, model, train_dataset)

    def batch_end_callback(trainer):
        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                prompt = "Once upon a time, "
                input_ids = encoder.encode(prompt)
                x = torch.tensor([input_ids], dtype=torch.long, device=trainer.device)

                y = model.generate(x, max_new_tokens=100, temperature=1.0, do_sample=True, top_k=40)

                generated_text = tokenizer.decode(y[0].tolist())
                print(f"\n{'=' * 40}")
                print(f"Sample at step {trainer.iter_num}:")
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated_text[len(prompt):]}")  # 只打印生成的部分
                print(f"{'=' * 40}\n")

            print("saving model...")
            chpt_path = os.path.join(config.system.work_dir, 'model.pt')
            torch.save(model.state_dict(), chpt_path)
            model.train()

    trainer.set_callback("on_batch_end", batch_end_callback)
    trainer.run()


