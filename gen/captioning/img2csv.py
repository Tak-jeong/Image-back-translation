import torch

import time
import argparse
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from util import get_dir, get_jpeg_list, root2csv, make_csv

parser = argparse.ArgumentParser(description='Arguments for processing.')

parser.add_argument('--out_path', 
                    default="/home/datasets/ImageNet1K/ILSVRC/Data/CLS-LOC/train_csv/",
                    type=str, 
                    help='path for output of data')

parser.add_argument('--root_path', 
                    default="/home/datasets/ImageNet1K/ILSVRC/Data/CLS-LOC/train",
                    type=str, 
                    help='path for root path of classes')

parser.add_argument('--file_format', 
                    default=r".JPEG",
                    type=str, 
                    help='file format of processing file')

parser.add_argument('--div_num', 
                    default=100,
                    type=int, 
                    help='num of processing classes')

parser.add_argument('--process_num', 
                    default=0,
                    type=int, 
                    help='what num you want to process bin?(maybe 0~9)')

parser.add_argument('--is_root',
                    action='store_true',
                    help='is it root folder for classes?(default: False)')


args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="large_coco", is_eval=True, device=device)

if __name__ == "__main__":
    if args.is_root:
        start = time.time()
        root2csv(args.root_path, args.out_path, args.file_format, model, vis_processors, device)
        print(f'Spend time for one class: {time.time() - start} sec')
    else:
        dirs = get_dir(args.root_path)
        classes = [dirs[i * args.div_num:(i + 1) * args.div_num] for i in range((len(dirs) + args.div_num - 1) // args.div_num )]
        for clas in tqdm(classes[args.process_num]):
            print(f'Num of picture in one class:{len(get_jpeg_list(clas, args.file_format))}')
            start = time.time()
            make_csv(clas, args.out_path, args.file_format, model, vis_processors, device)
            print(f'Spend time for one class: {time.time() - start} sec')