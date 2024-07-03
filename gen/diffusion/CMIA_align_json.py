# Model define & Initialize
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image

from typing import Callable, List, Optional, Union, Tuple

from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, DiffusionPipeline

import argparse

import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Dual Caption Dataset')

parser.add_argument('--root_folder', type=str, default="/home/tak/IBT/Image-back-translation/notebooks/submaterial", help='csv folder path')
parser.add_argument('--class1', type=str, default="n02134084", help='first image class name')
parser.add_argument('--class2', type=str, default="n04552348", help='second image class name')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_inference_steps', type=int, default=40, help='model inference steps')
parser.add_argument('--guidance_scale', type=float, default=1.5, help='model guidance scale')

args = parser.parse_args()


# model = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/sdxl-turbo",
#      torch_dtype=torch.float16, variant="fp16"
#      ).to(f"cuda")

base = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                                 torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")

refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0",
                                                    text_encoder_2=base.text_encoder_2,
                                                    vae=base.vae,
                                                    torch_dtype=torch.float16, 
                                                    use_safetensors=True,
                                                    variant="fp16").to("cuda")



class DualCaptionDataset(Dataset):
    def __init__(self, root_folder: Path, ensemble_class: List[str], save_image_folder_name: Path = Path("DualCaption_align_json_refine"), guidance_scale: float = args.guidance_scale, num_inference_steps: int = args.num_inference_steps):
        super().__init__()
        self.ensemble_class = ensemble_class
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.save_image_path = root_folder.joinpath(save_image_folder_name)
        self.save_image_path.mkdir(parents=True, exist_ok=True)

        self.json_folder = root_folder.joinpath('aligned_json')
        self.data1 = self.load_data(ensemble_class[0])
        self.data2 = self.load_data(ensemble_class[1])

    def load_data(self, class_name):
        json_path = self.json_folder.joinpath(f"{class_name}_modified.json")
        data = []
        with open(json_path, 'r') as file:
            for line in file:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(data)


    def __getitem__(self, idx):
        # 기존 코드
        prompt1 = self.data1['modified_caption'][idx % len(self.data1)]
        prompt2 = self.data2['modified_caption'][idx % len(self.data2)]
        prompt = f"mixture of {prompt1} and {prompt2}"

        # 이미지 경로 및 파일명 설정
        image_path1 = Path(self.data1['org_img_path'][idx % len(self.data1)])
        image_path2 = Path(self.data2['org_img_path'][idx % len(self.data2)])
        image_cls1 = image_path1.parts[-2]
        image_cls2 = image_path2.parts[-2]
        image_name1 = image_path1.stem
        image_name2 = image_path2.stem

        # 저장 경로 설정 및 폴더 생성
        save_path = self.save_image_path.joinpath(f"{image_cls1}_{image_cls2}", f"{image_name1}_{image_name2}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # generator_seed 설정
        generator_seed = torch.randint(0, 10000, (1,)).item()

        return {
            "prompt": prompt,
            "generator_seed": generator_seed,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps
        }, save_path

    def __len__(self):
        return min(len(self.data1), len(self.data2))
    
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

# for batch_idx, (data_batch, path_batch) in enumerate(tqdm(dataloader, desc="Generating Images")):
#     generator = [torch.Generator(device="cuda").manual_seed(seed.item()) for seed in data_batch["generator_seed"]]
#     data_batch["generator"] = generator
#     data_batch.pop("generator_seed")
#     data_batch["num_inference_steps"] = data_batch["num_inference_steps"][0].item()
#     images = model(**data_batch).images
#     for img, path in zip(images, path_batch):
#         save_path = Path(path)
#         save_folder = save_path.parent
#         save_folder.mkdir(parents=True, exist_ok=True)
#         img.save(save_path)
        
#         # image_folder = save_image_folder.joinpath(path[1][0] + "_" + path[1][1])
#         # image_folder.mkdir(parents=True, exist_ok=True)
#         # img.save(save_image_folder.joinpath(path[1][0] + "_" + path[1][1], path[2][0] + "_" + path[2][1] + ".png"))

# 배치 처리 루프
for batch_idx, (data_batch, path_batch) in enumerate(tqdm(dataloader, desc="Generating Images")):
    # Latent 이미지 생성을 위한 초기화
    latent_images = []

    # base 모델을 사용하여 latent 이미지 생성
    for idx, prompt in enumerate(data_batch["prompt"]):
        generator = torch.Generator(device="cuda").manual_seed(data_batch["generator_seed"][idx].item())
        latent_image = base(
            prompt=prompt,
            num_inference_steps=data_batch["num_inference_steps"][idx].item(),
            denoising_end=0.8,  # 필요에 따라 조절 가능
            output_type="latent",
            generator=generator
        ).images
        latent_images.append(latent_image)

    # refiner 모델을 사용하여 최종 이미지 생성
    final_images = []
    for idx, (latent_image, prompt) in enumerate(zip(latent_images, data_batch["prompt"])):
        final_image = refiner(
            prompt=prompt,
            num_inference_steps=data_batch["num_inference_steps"][idx].item(),
            denoising_start=0.8,  # 조절 가능
            image=latent_image,
            generator=torch.Generator(device="cuda").manual_seed(data_batch["generator_seed"][idx].item())
        ).images[0]
        final_images.append(final_image)

    # 최종 이미지 저장
    for img, path in zip(final_images, path_batch):
        save_path = Path(path)
        save_folder = save_path.parent
        save_folder.mkdir(parents=True, exist_ok=True)
        img.save(save_path)

