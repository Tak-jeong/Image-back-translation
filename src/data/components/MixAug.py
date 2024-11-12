# Custom dataset class
from typing import Any, Callable, Optional, Dict

from pathlib import Path
from PIL import Image

import os
import json
import random
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

from timm.data import Mixup

import torch
from torch.utils.data import Dataset
from . import CutMixUp
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from collections import defaultdict

import multiprocessing as mp


class MixAugDataset(datasets.ImageFolder):
    def __init__(self, root: str, cmia_path: str, btia_path: str,  mixup_args: Dict, cmia_prob=0.0, btia_prob=0.0, aug_num=1, sampling_ratio=1, **kwargs):
        super().__init__(root, **kwargs)
        self.cmia_path = Path(cmia_path)
        self.cmia_prob = cmia_prob

        self.btia_path = Path(btia_path)
        self.btia_prob = btia_prob

        self.aug_num = aug_num

        self.mixup_args = mixup_args
        self.mixup_fn = CutMixUp(**self.mixup_args)

        self.num_classes = len([name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))])

        default_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        self.transforms = kwargs.get('transform', default_transform)
        print("====================Augmentations====================")
        print(f"Using transform: {self.transforms}\n")
        print(f"Using cmia_prob: {self.cmia_prob}\n")
        print(f"Using btia_prob: {self.btia_prob}\n")
        print(f"Using mixup_args: {self.mixup_args}\n")
        print(f"Using aug_num: {self.aug_num}\n")
        print(f"Using sampling_ratio: {sampling_ratio}\n")
        print("=====================================================")

        # Organize samples by class
        samples_by_class = defaultdict(list)
        for path, target in self.samples:
            samples_by_class[target].append((path, target))
        
        # Select the first 10% from each class
        new_samples = []
        for target, samples in samples_by_class.items():
            samples.sort()  # Sort samples
            num_samples = int(len(samples) * sampling_ratio)
            new_samples.extend(samples[:num_samples])  # Select first 10%
        
        self.samples = new_samples


    def __getitem__(self, index):
        path, target = self.samples[index]
        return path, target


    def one_hot(self, x, num_classes, on_value=1., off_value=0.):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value) 


    def mixup_target(self, target, lam=1., smoothing=0.0):
        off_value = smoothing / self.num_classes
        on_value = 1. - smoothing + off_value
        y1 = self.one_hot(target, self.num_classes, on_value=on_value, off_value=off_value) 
        y2 = self.one_hot(target.flip(0), self.num_classes, on_value=on_value, off_value=off_value) 
        return y1 * lam + y2 * (1. - lam)


    def apply_cmia(self, paths, targets):
        if random.random() < self.cmia_prob:
            for i in range(0, len(paths), 2):  # Iterate over pairs of images
                # Get the class labels and file numbers for the selected images
                class1_name = Path(paths[i]).parent.name
                class2_name = Path(paths[i+1]).parent.name
                file_num1 = Path(paths[i]).stem.split('_')[-1]
                file_num2 = Path(paths[i+1]).stem.split('_')[-1]

                # Construct the CMIA image paths
                cmia_paths1 = glob(str(self.cmia_path / f'{class1_name}_{class2_name}' / f'{file_num1}_*.png'))
                cmia_path1 = cmia_paths1[0] if cmia_paths1 else paths[i]

                cmia_paths2 = glob(str(self.cmia_path / f'{class2_name}_{class1_name}' / f'{file_num2}_*.png'))
                cmia_path2 = cmia_paths2[0] if cmia_paths2 else paths[i+1]

                # Replace the original paths with the CMIA paths
                paths[i] = cmia_path1
                paths[i+1] = cmia_path2


                # Apply mixup only to the labels of the two images that we are augmenting
                targets[i], targets[i+1] = targets[i] * 0.5 + targets[i+1] * 0.5, targets[i+1] * 0.5 + targets[i] * 0.5


        return paths, targets



    def apply_btia(self, paths, targets):
        """Apply BTIA augmentation."""
        if random.random() < self.btia_prob:
            new_paths = []
            for path in paths:
                class_name = Path(path).parent.name
                file_num = Path(path).stem.split('_')[-1]
                
                # Find all augmented images corresponding to the original image, limited by k
                corresponding_images = [str(self.btia_path / class_name / f'{file_num}_{i}.png') for i in range(self.aug_num)]
                
                # Filter out non-existing paths
                corresponding_images = [img for img in corresponding_images if Path(img).exists()]

                if corresponding_images:
                    # Randomly choose one from the augmented images
                    new_path = random.choice(corresponding_images)
                else:
                    new_path = path  # if no corresponding image, keep the original
                
                new_paths.append(new_path)
            
            paths = new_paths
            
        return paths, targets


    def apply_mixup_cutmix(self, images, targets):
        """Apply mixup and cutmix augmentation."""
        images, targets = self.mixup_fn(images, targets)
        return images, targets


    def collate_fn(self, batch):
        assert len(batch) % 2 ==0, "batch size should be even when using this code"

        paths, targets = list(zip(*batch))
        paths = list(paths)
        targets = list(targets)
        
        # Convert labels to one-hot format
        targets = self.one_hot(torch.tensor(targets, dtype=torch.int64), self.num_classes)
        
        # Apply augmentations
        paths, targets = self.apply_cmia(paths, targets)
        paths, targets = self.apply_btia(paths, targets)
        
        images = [default_loader(path) for path in paths]

        # Apply transforms
        images = [self.transforms(image) for image in images]
        images = torch.stack(images)
        # resize_transform = Resize((512, 512))
        # images = [resize_transform(image) for image in images]
        # images = torch.stack([ToTensor()(image) for image in images])

        # Apply mixup and cutmix after loading and resizing the images
        images, targets = self.apply_mixup_cutmix(images, targets)

        return images, targets


class MixCutDataset(Dataset):
    def __init__(self, data_dir, wnid_to_idx, transform=None):
        self.data_dir = data_dir
        self.wnid_to_idx = wnid_to_idx
        self.transform = transform
        self.data = []
        self.targets = []
        self.prepare_data()

    def prepare_data(self):
        for cls_pair in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, cls_pair)
            if os.path.isdir(class_dir):
                try:
                    cls1, cls2 = cls_pair.split('_')
                    label1 = self.wnid_to_idx.get(cls1)
                    label2 = self.wnid_to_idx.get(cls2)
                    if label1 is None or label2 is None:
                        print(f"Invalid class labels in directory {cls_pair}: {cls1}, {cls2}")
                        continue

                    for file in os.listdir(class_dir):
                        if file.endswith('.JPEG'):
                            file_path = os.path.join(class_dir, file)
                            parts = file.split('_')
                            try:
                                if len(parts) >= 4 and 'lam' in parts[-2]:
                                    lam_parts = parts[-1].split('.')
                                    lam_str = f"{lam_parts[0]}.{lam_parts[1]}"
                                    lam = float(lam_str)
                                    self.data.append(file_path)
                                    self.targets.append((label1, label2, lam))
                                else:
                                    raise ValueError("Unexpected file format")
                            except (IndexError, ValueError) as e:
                                print(f"Error processing file {file} in directory {cls_pair}: {e} (parts={parts})")
                except Exception as e:
                    print(f"Error processing directory {cls_pair}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label1, label2, lam = self.targets[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, (label1, label2, lam)
    

    
