#from stable_diffusion.scripts.txt2img_custom import txt2img as sd
from diffusers import StableDiffusionXLPipeline
import json
import argparse
from multiprocessing import Manager
from multiprocessing.pool import Pool
from progress.bar import Bar
from tqdm import tqdm
from functools import partial
import os
#from pytorch_lightning import seed_everything
import shutil
import random
import pandas as pd 
import requests
import ast
import torch
from accelerate import PartialState



def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=100, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--n_samples', default=4, type=int,
                        help='Number of generated images.')
    parser.add_argument('--considered_images', default=-1, type=int,
                        help='Number of considered images.')
    parser.add_argument('--dataset', default=0, type=int,
                        help='Dataset (0: COCO;).')
    parser.add_argument('--generator', default=0, type=int,
                        help='Generator (0: Stable Diffusion;).')
    parser.add_argument('--list_file', default="../../datasets/coco/annotations/captions_train2014.json", type=str,
                        help='List of images.')
    parser.add_argument('--copy_files', default=False, action="store_true",
                        help='Do files copy')
    parser.add_argument('--gpu_id', default=-1, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--data_path', default="../../datasets/coco/train2014", type=str,
                        help='Path to data.')
    parser.add_argument('--excluded_images', default=["../../datasets/diffused_coco/train", "../../datasets/diffused_coco/val"], type=list,
                        help='Path to excluded images.')
    parser.add_argument('--output_path', default="..//datasets/sd_diffused_coco/train", type=str,
                        help='Output path.')
    opt = parser.parse_args()
    random.seed(42)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

    if opt.gpu_id == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = opt.gpu_id

    if opt.dataset == 0:
        f = open(opt.list_file)
        data = json.load(f)
        images = {}
        counter = 0
        for element in data["images"]:
            element = dict(element)
            id = element["id"]
            if id not in images:
                images[id] = os.path.join(opt.data_path, element["file_name"])

        captioned_images = {}
        for element in data["annotations"]:
            element = dict(element)
            id = element["image_id"]
            if id in images:
                captioned_images[id] = [images[id], element["caption"]]
            
        
        captioned_images = list(captioned_images.items())
        captioned_images = sorted(captioned_images, key=lambda k: random.random())
        if opt.considered_images != -1:
            captioned_images = captioned_images[:opt.considered_images]
            
        prompts = [row[1][1] for row in captioned_images]
        
        if opt.copy_files:
            for row in captioned_images:
                src_image = row[1][0]
                dst_path = os.path.join(opt.output_path, row[1][1])
                os.makedirs(dst_path, exist_ok = True)
                dst_image = os.path.join(dst_path, "real.png")        
                #shutil.copy(src_image, dst_image)
        


    if opt.generator == 0:
        txt2img = StableDiffusionXLPipeline.from_pretrained( "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
        txt2img.enable_vae_slicing()
        
        distributed_state = PartialState()
        txt2img.to(distributed_state.device)

    prompts = prompts * opt.n_samples
    BATCH_SIZE = 16
    batches = list(divide_chunks(prompts, BATCH_SIZE))      
    for index, batch_prompts in enumerate(batches):
        print(index, "/", len(batches))
        with distributed_state.split_between_processes(batch_prompts) as prompt:
            generated_images = txt2img(prompt)
            # TODO: Add images saving



    

