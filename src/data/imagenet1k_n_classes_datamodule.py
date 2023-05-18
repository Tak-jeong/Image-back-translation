from typing import Any, Dict, Optional, Tuple

import os
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from torchvision import datasets, transforms

import fnmatch

class ImageNet1K_N_ClassesDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, selected_classes=None, data_option='diff', train_real_folder='300_train',pin_memory=False, debug=False):
        super().__init__()
        self.data_dir = data_dir    # data_dir: ".data/ImageNet1K/ILSVRC/Data/CLS-LOC/"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.selected_classes = selected_classes
        self.data_option = data_option
        self.train_real_folder = train_real_folder
        self.pin_memory = pin_memory
        self.debug = debug

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _filter_dataset(self, dataset: datasets.ImageFolder):
        filtered_image_paths = []
        filtered_labels = []

        for image_path, label in dataset.imgs:
            class_name = image_path.split('/')[-2]
            if class_name in self.selected_classes:
                filtered_image_paths.append(image_path)
                filtered_labels.append(self.selected_classes.index(class_name))

        dataset.imgs = [(path, label) for path, label in zip(filtered_image_paths, filtered_labels)]
        dataset.samples = dataset.imgs
        dataset.targets = filtered_labels
        dataset.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.selected_classes)}
        dataset.classes = self.selected_classes

        return dataset




    

    def setup(self, stage: Optional[str] = None):
        train_real_path = os.path.join(self.data_dir, self.train_real_folder)

        if stage == 'fit' or stage is None:
            if self.data_option == 'diff' or self.data_option == 'concat':
                train_path = os.path.join(self.data_dir, 'train_diff')
            elif self.data_option == 'real':
                train_path = train_real_path

            full_train_dataset = datasets.ImageFolder(train_real_path, self.transform)
            self.train_dataset = self._filter_dataset(full_train_dataset)

            if self.data_option == 'diff' or self.data_option == 'concat':
                diff_train_dataset = datasets.ImageFolder(train_path, self.transform)
                filtered_image_paths = []

                for image_path, label in diff_train_dataset.imgs:
                    if check_suffix_0(image_path):
                        real_image_path = image_path.replace('train_diff', self.train_real_folder)
                        class_name = real_image_path.split('/')[-2]
                        file_name = class_name + '_' + real_image_path.split('/')[-1].split('_')[0] + '.JPEG'
                        real_image_path = os.path.dirname(real_image_path) + '/' + file_name

                        if real_image_path in [path for path, _ in self.train_dataset.imgs]:
                            filtered_image_paths.append(image_path)

                if not filtered_image_paths:
                    print(f"Selected class labels{self.selected_classes}")
                    raise ValueError("The filtered_image_paths is empty. Please check the dataset path and class labels.")

                parent_directory = os.path.dirname(os.path.dirname(filtered_image_paths[0]))
                diff_filtered_dataset = datasets.ImageFolder(parent_directory, self.transform)
                diff_filtered_dataset.imgs = [(path, label) for path, label in zip(filtered_image_paths, self.train_dataset.targets)]
                diff_filtered_dataset.samples = diff_filtered_dataset.imgs
                diff_filtered_dataset.targets = self.train_dataset.targets
                diff_filtered_dataset.class_to_idx = self.train_dataset.class_to_idx
                diff_filtered_dataset.classes = self.train_dataset.classes
                
                if self.data_option == 'concat':
                    self.train_dataset = ConcatDataset([self.train_dataset, diff_filtered_dataset])
                else:
                    self.train_dataset = diff_filtered_dataset

            full_val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'val_formatted'), self.transform)
            self.val_dataset = self._filter_dataset(full_val_dataset)

            if self.debug:
                self.train_dataset = torch.utils.data.Subset(self.train_dataset, range(0, len(self.train_dataset), 100))
                self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(0, len(self.val_dataset), 100))


        if stage == 'test' or stage is None:
            full_test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'), self.transform)
            self.test_dataset = self._filter_dataset(full_test_dataset)

            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

def check_suffix_0(file_path):
    file_name = os.path.basename(file_path)
    return fnmatch.fnmatch(file_name, '*_0.*')


if __name__ == "__main__":
    _ = ImageNet1K_N_ClassesDataModule()

