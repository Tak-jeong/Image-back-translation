# Model define & Initialize
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image

from typing import Callable, List, Optional, Union, Tuple

from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image

import argparse

parser = argparse.ArgumentParser(description='Dual Caption Dataset')

parser.add_argument('--root_folder', type=str, default="/mnt/nas65/Dataset/ImageNet1K/ILSVRC/Data/CLS-LOC", help='csv folder path')
parser.add_argument('--class1', type=str, default="n02134084", help='first image class name')
parser.add_argument('--class2', type=str, default="n04552348", help='second image class name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_inference_steps', type=int, default="10", help='model inference steps')
parser.add_argument('--guidance_scale', type=float, default="1.5", help='model guidance scale')

args = parser.parse_args()


model = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
     torch_dtype=torch.float16, variant="fp16"
     ).to(f"cuda")


# ImageCaptionPair Dataset
class ICP_Dataset(Dataset):
    def __init__(self, root_folder: Path, save_image_folder_name: Path, num_inference_steps: int=args.num_inference_steps, guidance_scale: float=args.guidance_scale):
        self.csv_list = list(root_folder.joinpath('aligned_csv').glob("*.csv"))
        self.save_image_path = root_folder.joinpath(save_image_folder_name)
        self.save_image_path.mkdir(parents=True, exist_ok=True)

        self.data = pd.concat([pd.read_csv(csv_file, sep=r",(?:(?!\s)+(?!')+(?!$))", engine='python') for csv_file in self.csv_list])
        self.image_path = self.data["org_img_path"].tolist()
        self.caption = self.data["caption"].tolist()

        if len(self.image_path) != len(self.caption):
            raise ValueError("image_path and caption must have the same length")

        # self.image_class = list(map(lambda x: x.split("/")[-2], self.image_path))
        # self.image_name = list(map(lambda x: x.split("/")[-1].split(".")[0], self.image_path))
        
        
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def __getitem__(self, idx):
        prompt = self.caption[idx]
        generator_seed = idx

        image_path = Path(self.image_path[idx])
        image_cls = image_path.parts[-2]
        image_name = image_path.stem

        return {"prompt": prompt, "generator_seed": generator_seed, "guidance_scale": self.guidance_scale, "num_inference_steps": self.num_inference_steps}, self.save_image_path.joinpath(image_cls).joinpath(image_name + ".png")

    def __len__(self):
        return len(self.data)
    

class DualCaptionDataset(ICP_Dataset):
    def __init__(self, root_folder: Path, ensemble_class: List[str], save_image_folder_name: Path = Path("DualCaption_align")):
        super().__init__(root_folder, save_image_folder_name)

        # ensemble class define
        if len(ensemble_class) != 2:
            raise ValueError("ensemble_class must be a list of two class")
        else:
            self.ensemble_class = ensemble_class

        # Split the dataframe by classes
        self.data1 = pd.concat([pd.read_csv(csv_file, sep=r",(?:(?!\s)+(?!')+(?!$))", engine='python') for csv_file in self.csv_list if csv_file.stem == self.ensemble_class[0]])
        self.data2 = pd.concat([pd.read_csv(csv_file, sep=r",(?:(?!\s)+(?!')+(?!$))", engine='python') for csv_file in self.csv_list if csv_file.stem == self.ensemble_class[1]])

        self.image_path1 = self.data1["org_img_path"].tolist()
        self.image_path2 = self.data2["org_img_path"].tolist()

        self.caption1 = self.data1["caption"].tolist()
        self.caption2 = self.data2["caption"].tolist()

        self.length = min(len(self.data1), len(self.data2))  # Use the minimum length of the two dataframes

    def __getitem__(self, idx):
        generator_seed = idx

        # Select one caption from each class
        prompt1 = self.caption1[idx % len(self.caption1)]  # Use modulo to prevent index out of range
        prompt2 = self.caption2[idx % len(self.caption2)]  # Use modulo to prevent index out of range

        image_path1 = Path(self.image_path1[idx % len(self.image_path1)])
        image_cls1 = image_path1.parts[-2]
        image_name1 = image_path1.stem.split("_")[-1]

        image_path2 = Path(self.image_path2[idx % len(self.image_path2)])
        image_cls2 = image_path2.parts[-2]
        image_name2 = image_path2.stem.split("_")[-1]

        prompt = f"mixture of {prompt1} and {prompt2}"
        # image_path, image_cls, image_name = [image_path1, image_path2], [image_cls1, image_cls2], [image_name1, image_name2]

        save_path = self.save_image_path.joinpath(f"{image_cls1}_{image_cls2}/{image_name1}_{image_name2}.png")

        return {"prompt": prompt, "generator_seed": generator_seed, "guidance_scale": self.guidance_scale,"num_inference_steps": self.num_inference_steps}, save_path

    def __len__(self):
        return self.length
    
def collate_fn(batch):
    # 데이터 및 메타데이터 리스트 분리
    data_list, meta_list = zip(*batch)

    # 데이터 배치 생성
    data_batch = {
        'prompt': [item['prompt'] for item in data_list],
        'generator_seed': torch.tensor([item['generator_seed'] for item in data_list]),
        'guidance_sclae': torch.tensor([item['guidance_scale'] for item in data_list]),
        'num_inference_steps': torch.tensor([item['num_inference_steps'] for item in data_list])
    }

    # 메타데이터는 리스트로 유지
    meta_batch = list(meta_list)

    return data_batch, meta_batch



root_folder = Path(args.root_folder)

dataset = DualCaptionDataset(root_folder, ensemble_class=[args.class1, args.class2])
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

for batch_idx, (data_batch, path_batch) in enumerate(dataloader):
    generator = [torch.Generator(device="cuda").manual_seed(seed.item()) for seed in data_batch["generator_seed"]]
    data_batch["generator"] = generator
    data_batch.pop("generator_seed")
    data_batch["num_inference_steps"] = data_batch["num_inference_steps"][0].item()
    images = model(**data_batch).images
    for img, path in zip(images, path_batch):
        save_path = Path(path)
        save_folder = save_path.parent
        save_folder.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        
        # image_folder = save_image_folder.joinpath(path[1][0] + "_" + path[1][1])
        # image_folder.mkdir(parents=True, exist_ok=True)
        # img.save(save_image_folder.joinpath(path[1][0] + "_" + path[1][1], path[2][0] + "_" + path[2][1] + ".png"))

