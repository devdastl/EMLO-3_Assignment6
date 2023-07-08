import tiktoken
import requests
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import itertools

from einops import rearrange, reduce, repeat, einsum
from einops.layers.torch import Rearrange, Reduce

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning import LightningModule, LightningDataModule
from torchmetrics import MaxMetric, MeanMetric, MinMetric

from typing import Any, Optional

class HarryPotterDataset(Dataset):
    def __init__(self, data_dir="./data", block_size=8, download=True):
        super().__init__()

        self.block_size = block_size
        self.data_dir = data_dir
        res = []
        book_links = ["https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt",
                      "https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%202%20-%20The%20Chamber%20of%20Secrets.txt",
                     "https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%203%20-%20The%20Prisoner%20of%20Azkaban.txt",
                    "https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%204%20-%20The%20Goblet%20of%20Fire.txt",
                    "https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%205%20-%20The%20Order%20of%20the%20Phoenix.txt",
                    "https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%206%20-%20The%20Half%20Blood%20Prince.txt",
                    "https://github.com/formcept/whiteboard/blob/master/nbviewer/notebooks/data/harrypotter/Book%207%20-%20The%20Deathly%20Hallows.txt"
                    ]

        if download:
            for book in book_links:
                res.append(requests.get(book, allow_redirects=True))

           # res = requests.get(
           #     "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
           #     allow_redirects=True
           # )

            Path(self.data_dir).mkdir(exist_ok=True, parents=True)

            with (Path(self.data_dir) / "data.txt").open("wb") as f:
                for cont in res:
                    f.write(cont.content)
                    f.write(b"\n")
                #f.write(res.content)

        with (Path(self.data_dir) / "data.txt").open() as f:
            self.text = f.read()

        cl100k_base = tiktoken.get_encoding("cl100k_base")

        # In production, load the arguments directly instead of accessing private attributes
        # See openai_public.py for examples of arguments for specific encodings
        self.encoder = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<|im_start|>": 100264,
                "<|im_end|>": 100265,
            }
        )

        self.data = np.array(self.encoder.encode(self.text))

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(
            (self.data[idx:idx + self.block_size]).astype(np.int64)
        )

        y = torch.from_numpy(
            (self.data[idx+1:idx+1+self.block_size]).astype(np.int64)
        )

        return x, y


class HarryPotterDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_ratio=0.7,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        block_size=8,
        download=True
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        HarryPotterDataset(self.hparams.data_dir, download=self.hparams.download, block_size=self.hparams.block_size)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dataset = HarryPotterDataset(self.hparams.data_dir, download=False, block_size=self.hparams.block_size)

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[self.hparams.train_ratio, (1-self.hparams.train_ratio)],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )