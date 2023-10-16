import os
import torch
from PIL import Image
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
    
    for file in get_jpeg_list(path,file_format):
        w_path = os.path.join(path, file)
        raw_image = Image.open(w_path).convert('RGB')
        
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        text = model.generate({"image": image})
        
        f.write(w_path + "," + text[0] + "\n")
    f.close()

def root2csv(root_dir, out_path, file_format, model, vis_processors, device):
    for dir in os.scandir(root_dir):
        if dir.is_dir():
            make_csv(dir.path, out_path, file_format, model, vis_processors, device)