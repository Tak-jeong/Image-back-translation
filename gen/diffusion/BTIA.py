import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionPipeline

import googletrans
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='en-ko-en Dataset')
parser.add_argument('--model', choices=range(1,5), default="1", help='model name')
parser.add_argument('--root_path', type=str, default="/data1/ImageNet1K/Annotations/Data/CLS-LOC/", help='csv folder path')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--class_list', nargs='+', help='List of classes to process')

args = parser.parse_args()


model = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-"+ str(args.model), 
	use_auth_token=True
).to("cuda")


# ImageCaptionPair Dataset
class ICP_Dataset(Dataset):
    def __init__(self, root_folder: Path, save_image_folder_name: Path):
        self.csv_list = list(root_folder.joinpath('train_csv').glob("*.csv"))
        self.save_image_path = root_folder.joinpath(save_image_folder_name)
        self.save_image_path.mkdir(parents=True, exist_ok=True)

        self.data = pd.concat([pd.read_csv(csv_file, sep=r",(?:(?!\s)+(?!')+(?!$))", engine='python') for csv_file in self.csv_list])
        
        if args.class_list:
            self.data = self.data[self.data['org_img_path'].str.contains('|'.join([fr'\b{cls}\b' for cls in args.class_list]))]

        self.image_path = self.data["org_img_path"].tolist()
        self.caption = self.data["caption"].tolist()

        if len(self.image_path) != len(self.caption):
            raise ValueError("image_path and caption must have the same length")

        # self.image_class = list(map(lambda x: x.split("/")[-2], self.image_path))
        # self.image_name = list(map(lambda x: x.split("/")[-1].split(".")[0], self.image_path))

    def __getitem__(self, idx):
        prompt = self.caption[idx]
        generator_seed = idx
        num_inference_steps = 20

        image_path = Path(self.image_path[idx])
        image_cls = image_path.parts[-2]
        image_name = image_path.stem

        return {"prompt": prompt, "generator_seed": generator_seed, "num_inference_steps": num_inference_steps}, self.save_image_path.joinpath(image_cls).joinpath(image_name + ".png")

    def __len__(self):
        return len(self.data)


# Back Translated Image Dataset
class BTI_Dataset(ICP_Dataset):
    def __init__(self, root_folder: Path, save_image_folder_name: Path = Path("en-ko-en"), src_lang: str = 'en', tgt_lang: str = 'ko'):
        super().__init__(root_folder, save_image_folder_name)
        self.translator = googletrans.Translator()
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        
    
    def __getitem__(self, idx):
        prompt = self.caption[idx]
        trans_prompt = self.translator.translate(prompt, dest=self.tgt_lang, src=self.src_lang).text
        back_trans_prompt = self.translator.translate(trans_prompt, dest=self.src_lang, src=self.tgt_lang).text
        generator_seed = idx
        num_inference_steps = 20

        image_path = Path(self.image_path[idx])
        image_cls = image_path.parts[-2]
        image_name = image_path.stem

        return {"prompt": back_trans_prompt, "generator_seed": generator_seed, "num_inference_steps": num_inference_steps}, self.save_image_path.joinpath(image_cls).joinpath(image_name + ".png")

def collate_fn(batch):
    # 데이터 및 메타데이터 리스트 분리
    data_list, meta_list = zip(*batch)

    # 데이터 배치 생성
    data_batch = {
        'prompt': [item['prompt'] for item in data_list],
        'generator_seed': torch.tensor([item['generator_seed'] for item in data_list]),
        'num_inference_steps': torch.tensor([item['num_inference_steps'] for item in data_list]),
    }

    # 메타데이터는 리스트로 유지
    meta_batch = list(meta_list)

    return data_batch, meta_batch

root_path = Path(args.root_path)
dataset = BTI_Dataset(root_path)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

for batch_idx, (data_batch, save_path_batch) in enumerate(dataloader):
    generator = [torch.Generator(device="cuda").manual_seed(seed.item()) for seed in data_batch["generator_seed"]]
    data_batch["generator"] = generator
    data_batch.pop("generator_seed")
    data_batch["num_inference_steps"] = data_batch["num_inference_steps"][0].item()

    # 먼저 이미지가 이미 존재하는지 확인
    to_generate = []
    for save_path in save_path_batch:
        if not os.path.exists(save_path):
            to_generate.append(True)
        else:
            to_generate.append(False)

    # 존재하지 않는 이미지만 생성
    if any(to_generate):
        images = model(**data_batch).images
    else:
        continue

    # 존재하지 않는 이미지만 저장
    for img, save_path, generate in zip(images, save_path_batch, to_generate):
        if generate:
            save_path = Path(save_path)
            image_folder = save_path.parent
            image_folder.mkdir(parents=True, exist_ok=True)
            img.save(save_path)


# for batch_idx, (data_batch, save_path_batch) in enumerate(dataloader):
#     generator = [torch.Generator(device="cuda").manual_seed(seed.item()) for seed in data_batch["generator_seed"]]
#     data_batch["generator"] = generator
#     data_batch.pop("generator_seed")
#     data_batch["num_inference_steps"] = data_batch["num_inference_steps"][0].item()
#     images = model(**data_batch).images
#     for img, save_path in zip(images, save_path_batch):
#         save_path = Path(save_path)
#         if not os.path.exists(save_path):
#             image_folder = save_path.parent
#             image_folder.mkdir(parents=True, exist_ok=True)
#             img.save(save_path)

