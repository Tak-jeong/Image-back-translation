#!/bin/sh

find  /home/tak/IBT/Image-back-translation/data/ImageNet1K/ILSVRC/Data/CLS-LOC/train -type d | while read -r dir
do
        printf "%s:\t" "$dir";
        find "$dir" -type f | wc -l;
done