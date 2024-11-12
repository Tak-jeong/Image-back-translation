# Custom dataset class
import os
from PIL import Image
from pathlib import Path
from typing import Any, Callable, Optional, Dict
from torch.utils.data import Dataset


class CapMixDataset(Dataset):
    def __init__(self, data_dir, class_to_idx, transform=None):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
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
                    label1 = self.class_to_idx.get(cls1)
                    label2 = self.class_to_idx.get(cls2)
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
    