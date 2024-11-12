from .components import ImageNetDataset, MixCutDataset, SDDataset

import os
from pathlib import Path
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.datasets as datasets

def mixcut_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    
    label1, label2, lam = zip(*batch)
    if len(label1) > 0 and len(label2) > 0 and len(lam) > 0:
        label1 = torch.tensor(label1, dtype=torch.long)
        label2 = torch.tensor(label2, dtype=torch.long)
        lam = torch.tensor(lam, dtype=torch.float)
        targets = (label1, label2, lam)
    else:
        targets = torch.tensor(targets, dtype=torch.long)
    
    return images, targets

def unified_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    label1, label2, lam = zip(*targets)
    label1 = torch.tensor(label1, dtype=torch.long)
    label2 = torch.tensor(label2, dtype=torch.long)
    lam = torch.tensor(lam, dtype=torch.float)
    return images, (label1, label2, lam)

class CapMixDataModule(LightningDataModule):
    def __init__(self, root_path: str, aug_folder: str, train_folder='TE_10', use_caption_ratio=False, caption_folder='te_json_aligned' , batch_size=32, num_workers=4):
        super().__init__()
        self.root_path = root_path
        self.aug_true = aug_folder not in ['None', 'none']
        self.aug_folder = aug_folder
        if self.aug_true:
            self.mixcut_path = os.path.join(self.root_path, '10way', aug_folder)
        
        self.use_caption_ratio = use_caption_ratio
        self.caption_path = Path(self.root_path).parent.joinpath('captions')
        self.caption_folder = caption_folder

        self.train_folder = train_folder
        self.train_path = os.path.join(self.root_path, '10way', self.train_folder)
        self.val_path = os.path.join(self.root_path, 'whole', 'val_formatted')

        self.wnid = [wnid for wnid in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, wnid))]
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnid)}
        self.batch_size = batch_size
        self.num_workers = num_workers

        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor()
        # ])
        
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def setup(self, stage=None):
        if self.aug_true:
            if self.aug_folder.lower() in ['cutmix', 'mixup']:
                self.imagenet_dataset = ImageNetDataset(self.train_path, self.wnid_to_idx, transform=self.transform)
                self.capmix_dataset = MixCutDataset(self.mixcut_path, self.wnid_to_idx, transform=self.transform)
                self.train_dataset = ConcatDataset([self.imagenet_dataset, self.capmix_dataset])

            elif self.aug_folder.lower() in ['te_cutmix', 'te_mixup']:
                self.imagenet_dataset = datasets.Imagenette(root='~/.datasets', split='train', transform=self.transform)
                self.capmix_dataset = MixCutDataset(self.mixcut_path, self.imagenet_dataset.wnid_to_idx, transform=self.transform)
                self.train_dataset = ConcatDataset([self.imagenet_dataset, self.capmix_dataset])

            else:
                self.imagenet_dataset = ImageNetDataset(self.train_path, self.wnid_to_idx, transform=self.transform)
                self.diffusion_dataset = SDDataset(self.mixcut_path, self.wnid_to_idx, self.caption_path, self.caption_folder, self.use_caption_ratio, transform=self.transform)
                self.train_dataset = ConcatDataset([self.imagenet_dataset, self.diffusion_dataset])
        else:
            if self.caption_folder.lower() in ['te_json_aligned', 'te_json']:
                self.train_dataset = datasets.Imagenette(root='~/.datasets', split='train', transform=self.transform)
            else:
                self.train_dataset = ImageNetDataset(self.train_path, self.wnid_to_idx, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=unified_collate_fn)

    def val_dataloader(self):
        if self.caption_folder.lower() in ['te_json_aligned', 'te_json']:
            val_dataset = datasets.Imagenette(root='~/.datasets', split='val', transform=self.transform)

        else:
            val_dataset = ImageNetDataset(self.val_path, self.wnid_to_idx, transform=self.transform)

        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)