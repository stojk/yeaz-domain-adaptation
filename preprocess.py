# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:23:25 2021

@author: gligorov
"""

import glob
import os
import os.path as op
import shutil
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm


def crop_into_patches(
    src_path: str,
    dst_path: str,
    variance_threshold: int,
    patch_size: int = 256
):
    """Crops the images into patches of size patch_size x patch_size and saves them in the
    Patches folder. The patches are saved as patch_i.tif, where i is the index of the patch.
    
    Arguments:
        src_path: path to the folder containing the images
        dst_path: path to the folder where the patches will be saved
        variance_threshold: empirical varience threshold
        patch_size: size of the patches
    """

    # Delete the dst_path if it exists
    if op.exists(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    half_size = patch_size//2
    crop_list = []
    image_list = [
        Image.open(filename) for filename in glob.glob(op.join(src_path,'*.tif'))
    ]

    for image in image_list:
        #we take the overlapping fragments
        m = int(np.floor(image.size[0]/half_size)) 
        n = int(np.floor(image.size[1]/half_size))

        for i in range(m-1):
            for j in range(n-1):
                left = i*half_size
                right = i*half_size+patch_size
                top = j*half_size
                bottom = j*half_size+patch_size
                crop_list.append(image.crop((left, top, right, bottom)))

    for i, patch in enumerate(tqdm(crop_list)):
        #empirical threshold that works very well
        if np.var(patch) > variance_threshold:
            patch.save(op.join(dst_path, f"patch_{i}.tif"))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--src_path', type=str, required=True,
                        help='Path to the folder containing the images')
    parser.add_argument('--dst_path', type=str, required=True,
                        help='Path to the folder where the patches will be saved')
    parser.add_argument('--var_thr', type=int,
                        required=True, help='Empirical varience threshold')
    parser.add_argument('--patch_size', type=int,
                        default=256, help='Size of the patches')

    args = parser.parse_args()
    crop_into_patches(
        src_path=args.src_path,
        dst_path=args.dst_path,
        variance_threshold=args.var_thr,
        patch_size=args.patch_size
    )
