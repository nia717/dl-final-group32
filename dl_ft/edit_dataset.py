from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import os 
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        # with open(Path(self.path, "seeds.json")) as f:
        #     self.seeds = json.load(f)
        with open(os.path.join(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)
            
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        # 修复路径处理，确保使用正确的路径分隔符
        # 将Windows风格的路径转换为当前操作系统的路径格式
        name = name.replace('\\', os.path.sep)
        propt_dir = os.path.join(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]

        # 检查目录是否存在
        if not os.path.exists(propt_dir):
            print(f"Warning: Directory does not exist: {propt_dir}")
            # 尝试使用替代路径
            alt_name = name.split('\\')[-1] if '\\' in name else name
            alt_propt_dir = os.path.join(self.path, alt_name)
            if os.path.exists(alt_propt_dir):
                propt_dir = alt_propt_dir
                print(f"Using alternative path: {propt_dir}")

        try:
            prompt_path = os.path.join(propt_dir, "prompt.json")
            with open(prompt_path) as fp:
                prompt = json.load(fp)["edit"]
        except FileNotFoundError:
            print(f"Warning: Could not find {prompt_path}")
            # 提供一个默认值而不是抛出错误
            return self.__getitem__((i + 1) % len(self.seeds))  # 尝试下一个样本
        # Try to open images with the pattern "{seed}_0.jpg" and "{seed}_1.jpg"
        # If that fails, try to open images with the pattern "0.jpg" and "1.jpg"
        try:
            image_0 = Image.open(os.path.join(propt_dir, f"{seed}_0.jpg"))
            image_1 = Image.open(os.path.join(propt_dir, f"{seed}_1.jpg"))
        except FileNotFoundError:
            image_0 = Image.open(os.path.join(propt_dir, "0.jpg"))
            image_1 = Image.open(os.path.join(propt_dir, "1.jpg"))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(edited=image_1, edit=dict(c_concat=image_0, c_crossattn=prompt))


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.res = res

        # with open(Path(self.path, "seeds.json")) as f:
        #     self.seeds = json.load(f)

        with open(os.path.join(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)
            
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        # 修复路径处理，确保使用正确的路径分隔符
        name = name.replace('\\', os.path.sep)
        propt_dir = os.path.join(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]

        # 检查目录是否存在
        if not os.path.exists(propt_dir):
            print(f"Warning: Directory does not exist: {propt_dir}")
            # 尝试使用替代路径
            alt_name = name.split('\\')[-1] if '\\' in name else name
            alt_propt_dir = os.path.join(self.path, alt_name)
            if os.path.exists(alt_propt_dir):
                propt_dir = alt_propt_dir
                print(f"Using alternative path: {propt_dir}")

        try:
            prompt_path = os.path.join(propt_dir, "prompt.json")
            with open(prompt_path) as fp:
                prompt = json.load(fp)
                edit = prompt["edit"]
                input_prompt = prompt["input"]
                output_prompt = prompt["output"]
        except FileNotFoundError:
            print(f"Warning: Could not find {prompt_path}")
            # 提供一个默认值而不是抛出错误
            return self.__getitem__((i + 1) % len(self.seeds))  # 尝试下一个样本

        # Try to open image with the pattern "{seed}_0.jpg"
        # If that fails, try to open image with the pattern "0.jpg"
        try:
            image_0 = Image.open(os.path.join(propt_dir, f"{seed}_0.jpg"))
        except FileNotFoundError:
            image_0 = Image.open(os.path.join(propt_dir, "0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)
