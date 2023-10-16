import os
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path


def get_jpeg_list(Dir, file_format):
    JPEG_list = [file for file in os.listdir(Dir) if file.endswith(file_format)]
    return JPEG_list

def get_format_list(Dir, file_format):
    format_list = [os.path.join(Dir,file) for file in os.listdir(Dir) if file.endswith(file_format)]
    return format_list


def get_dir(root_dir):
    dirs = []
    for dir in os.scandir(root_dir):
        if dir.is_dir():
            dirs.append(dir.path)
    return dirs



def make_csv(path, out_path, file_format, model, vis_processors, device):
    f = open(out_path + str(path).split("/")[-1] + ".csv","w",encoding="UTF-8")
    f.write("org_img_path,caption\n")
    
    # Create a transform pipeline for data augmentation
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])
    
    for file in get_jpeg_list(path, file_format):
        w_path = os.path.join(path, file)
        raw_image = Image.open(w_path).convert('RGB')
        
        # Apply the data augmentation transformations
        augmented_image = augment_transform(raw_image)
        
        # Apply the existing vis_processors for additional preprocessing
        image = vis_processors["eval"](augmented_image).unsqueeze(0).to(device)
        
        # Generate caption
        text = model.generate({"image": image})
        
        f.write(w_path + "," + text[0] + "\n")
    f.close()

def root2csv(root_dir, out_path, file_format, model, vis_processors, device):
    for dir in os.scandir(root_dir):
        if dir.is_dir():
            make_csv(dir.path, out_path, file_format, model, vis_processors, device)