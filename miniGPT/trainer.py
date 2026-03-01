import time
from collections import defaultdict
from dataclasses import replace

import torch
from torch.utils.data import DataLoader
from .utils import CfgNode

class Trainer:
    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.device = "auto"
        C.num_workers = 4
        C.max_iters = 5
        C.batch_size = 16
        C.lr = 3e-4
        C.betas = (0.9, 0.999)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0

        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = self.model.to(self.device)
        print(f"on {self.device}")

        self.iter_num = 0 # 当前训练步数
        self.iter_time = 0.0 # 时间基准点
        self.iter_dt = 0.0 # 单步耗时 一个batch

    def add_callback(self, onevent, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callback(self, onevent):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizers(config)

        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=int(1e10),
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = [t.to(self.device) for t in batch]

            x, y = batch

            logits, self.loss = model(x, y)
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callback('on_batch_end')
            self.iter_num += 1
            time_now = time.time()
            self.iter_dt = time_now - self.iter_time
            self.iter_time = time_now

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break




