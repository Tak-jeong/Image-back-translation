from typing import Any, Callable, Optional, Dict
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset
from PIL import Image
import os

import multiprocessing as mp


class ImageNetDataset(Dataset):
    def __init__(self,data_dir, class_to_idx, transform=None):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.data = []
        self.targets = []
        self.prepare_data()

    def prepare_data(self):
        for cls in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, cls)
            if os.path.isdir(class_dir) and cls in self.class_to_idx:
                for file in os.listdir(class_dir):
                    if file.endswith('.JPEG'):
                        self.data.append(os.path.join(class_dir, file))
                        self.targets.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.targets[idx]
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, (label, label, 1.0)
    

class SDDataset(Dataset):
    def __init__(self, data_dir: str, class_to_idx: Dict, caption_path: Path, caption_folder: str, use_caption_ratio: bool,
                 transform=None):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.caption_path = caption_path.joinpath(caption_folder)
        self.transform = transform
        self.use_caption_ratio = use_caption_ratio
        self.data = []
        self.targets = []
        self.captions = {}
        self.prepare_data()

    def prepare_data(self):
        if self.use_caption_ratio:
            self.load_all_captions()
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(self.process_class_pair, os.listdir(self.data_dir))
        
        for result in results:
            if result:
                self.data.extend(result[0])
                self.targets.extend(result[1])

    def load_all_captions(self):
        for caption_file in self.caption_path.glob('*_modified.json'):
            cls = caption_file.stem.split('_')[0]
            with open(caption_file, 'r') as f:
                self.captions[cls] = {json.loads(line)['org_img_path'].split('/')[-1].split('.')[0]: json.loads(line)['caption'] for line in f}

    def process_class_pair(self, cls_pair):
        class_dir = os.path.join(self.data_dir, cls_pair)
        if not os.path.isdir(class_dir):
            return None

        try:
            cls1, cls2 = cls_pair.split('_')
            label1, label2 = self.class_to_idx.get(cls1), self.class_to_idx.get(cls2)

            data = []
            targets = []

            for file in os.listdir(class_dir):
                if not file.endswith('.png'):
                    continue

                file_path = os.path.join(class_dir, file)

                if self.use_caption_ratio:
                    file1, file2 = self.split_filename(file)
                    caption1 = self.find_caption(cls1, file1)
                    caption2 = self.find_caption(cls2, file2)
                    if not caption1 or not caption2:
                        print(f"Error processing file {file} in directory {cls_pair}: No matching caption found")
                        continue
                    lam = len(caption1) / (len(caption1) + len(caption2))
                else:
                    lam = 0.5

                data.append(file_path)
                targets.append((label1, label2, lam))

            return data, targets
        except Exception as e:
            print(f"Error processing directory {cls_pair}: {e}")
            return None
        
    def find_caption(self, cls, file_part):
        for key, caption in self.captions[cls].items():
            if file_part in key or key in file_part:
                return caption
        return None

    def split_filename(self, filename):
        parts = filename.split('.')[0].split('_')
        
        # Find all parts that could be the start of a filename
        potential_splits = [i for i, part in enumerate(parts) if part.startswith('n') and part[1:].isdigit() or part.startswith('ILSVRC')]
        
        if len(potential_splits) == 2:
            # If we found exactly two potential starting points, use these to split
            return '_'.join(parts[:potential_splits[1]]), '_'.join(parts[potential_splits[1]:])
        elif len(potential_splits) > 2:
            # If we found more than two, use the middle one to split
            mid = len(potential_splits) // 2
            return '_'.join(parts[:potential_splits[mid]]), '_'.join(parts[potential_splits[mid]:])
        else:
            # If we couldn't find two clear split points, fall back to splitting in the middle
            mid = len(parts) // 2
            return '_'.join(parts[:mid]), '_'.join(parts[mid:])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label1, label2, lam = self.targets[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        return img, (label1, label2, lam)