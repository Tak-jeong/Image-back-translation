import os
from pathlib import Path
from typing import Dict



from .components import MixAugDataset

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

class MixAugDataModule(LightningDataModule):
    def __init__(self, root_path: str, train_dir: str,val_dir: str, mixup_args: Dict, mixaug_args: Dict, batch_size: int = 64):
        super().__init__()
        self.root_path = root_path
        self.train_dir = os.path.join(self.root_path, train_dir)

        self.val_dir = os.path.join(self.root_path, val_dir)
        # self.test_dir = os.path.join(self.root_path, 'test')

        self.mixup_args = mixup_args
        self.mixaug_args = mixaug_args

        self.cmia_dir = os.path.join(self.root_path, self.mixaug_args['cmia_dir'])
        self.cmia_prob = self.mixaug_args['cmia_prob']
        self.btia_dir = os.path.join(self.root_path, self.mixaug_args['btia_dir'])
        self.btia_prob = self.mixaug_args['btia_prob']
        
        self.aug_num = self.mixaug_args['aug_num']

        self.batch_size = batch_size

        self.train_transform = transforms.Compose([
                                transforms.Resize((512, 512)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225]),
                                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                                    ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])

    def one_hot(self, x, num_classes, on_value=1., off_value=0.):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)
    
    def val_collate_fn(self, batch):
        images, targets = list(zip(*batch))
        targets = self.one_hot(torch.tensor(targets, dtype=torch.int64), self.train_dataset.num_classes)
        images = torch.stack(images)
        return images, targets




    def setup(self, stage=None):
        # Split the dataset into train, val, and test sets
        # Create instances of the CustomDataset for each split
        self.train_dataset = MixAugDataset(root=self.train_dir, cmia_path=self.cmia_dir, btia_path=self.btia_dir,mixup_args=self.mixup_args, transform = self.train_transform, cmia_prob=self.cmia_prob, btia_prob=self.btia_prob, aug_num=self.aug_num)
        self.val_dataset = datasets.ImageFolder(self.val_dir, self.val_transform)
        # self.test_dataset = datasets.ImageFolder(self.test_dir, self.val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=self.train_dataset.collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=self.val_collate_fn)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=self.val_collate_fn)
