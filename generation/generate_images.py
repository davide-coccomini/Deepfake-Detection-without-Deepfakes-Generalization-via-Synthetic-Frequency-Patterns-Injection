#from stable_diffusion.scripts.txt2img_custom import txt2img as sd
import json
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
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
import gc
import numpy as np
from PIL import Image

gc.collect()

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--n_samples', default=1, type=int,
                        help='Number of generated images.')
    parser.add_argument('--considered_images', default=-1, type=int,
                        help='Number of considered images.')
    parser.add_argument('--dataset', default=0, type=int,
                        help='Dataset (0: COCO;).')
    parser.add_argument('--generator', default=0, type=int,
                        help='Generator (0: Stable Diffusion XL; 1: Stable Diffusion V2.1; 2: Stable Diffusion V2).')
    parser.add_argument('--list_file', default="../../data/coco/annotations/captions_train2017.json", type=str,
                        help='List of images.')
    parser.add_argument('--copy_files', default=False, action="store_true",
                        help='Do files copy')
    parser.add_argument('--regenerate', default=False, action="store_true",
                        help='Regenerated already generated images.')
    parser.add_argument('--gpu_id', default=-1, type=int,
                        help='ID of GPU to be used.')
    parser.add_argument('--data_path', default="../../data/coco/train/train2017", type=str,
                        help='Path to data.')
    parser.add_argument('--excluded_images', default=["../../data/diffused_coco/train", "../../data/diffused_coco/train"], type=list,
                        help='Path to excluded images.')
    parser.add_argument('--output_path', default="../../data/sd_diffused_coco/train", type=str,
                        help='Output path.')
    opt = parser.parse_args()
    random.seed(42)


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
        c = 0
        for element in data["annotations"]:
            c += 1
            element = dict(element)
            id = element["image_id"]
            if id in images:
                if id in captioned_images:
                    captioned_images[id][1].append(element["caption"])
                else:
                    captioned_images[id] = [images[id], [element["caption"]]]
        

        # Filter only one caption per image
        for id in captioned_images:
            captioned_images[id][1] = random.choice(captioned_images[id][1])

        # Filter only captions for which a generation has not been done
        if not opt.regenerate:
            rows_to_remove = []
            for id, (image_path, caption) in captioned_images.items():
                dst_path = os.path.join(opt.output_path, caption)
                if os.path.exists(dst_path) and len(os.listdir(dst_path)) > 1:
                    rows_to_remove.append(id)
            
            for id in rows_to_remove:
                del captioned_images[id]

            print("Ignored", len(rows_to_remove), "captions already generated.")

                
   
        captioned_images = list(captioned_images.items())
        captioned_images = sorted(captioned_images, key=lambda k: random.random())
        if opt.considered_images != -1:
            captioned_images = captioned_images[:opt.considered_images]
        
        all_prompts = [row[1][1] for row in captioned_images]
        
        if opt.copy_files:
            print("Copying real images...")
            for index, row in enumerate(captioned_images):
                if index % 10000 == 0:
                    print("Copied", index, "/", len(captioned_images))
                src_image = row[1][0]
                dst_path = os.path.join(opt.output_path, row[1][1])
                os.makedirs(dst_path, exist_ok = True)
                dst_image = os.path.join(dst_path, "real.png")    
                if os.path.exists(dst_image):
                    continue    
                shutil.copy(src_image, dst_image)


        if opt.generator == 0:
            txt2img = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,  
                local_files_only=True,
                use_safetensors=True
            )
        elif opt.generator == 1:
            txt2img = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-base",
                torch_dtype=torch.float16,
                local_files_only=True, 
                use_safetensors=True
            )
        elif opt.generator == 2:
            txt2img = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2",
                torch_dtype=torch.float16, 
                local_files_only=False,
                use_safetensors=True
            )

        txt2img.enable_vae_slicing()
        
        distributed_state = PartialState()
        txt2img.to(distributed_state.device)

        with torch.no_grad():
            all_prompts = all_prompts * opt.n_samples
            BATCH_SIZE = 64
            batches = list(divide_chunks(all_prompts, BATCH_SIZE))
            print("Prompts:", len(all_prompts))
            counter = 0    
            for index, batch_prompts in enumerate(batches):
                print(index, "/", len(batches))
                with distributed_state.split_between_processes(batch_prompts) as prompts:
                    generated_images = txt2img(prompts, num_inference_steps=100)["images"]
                    for i in range(len(prompts)):
                        dst_path = os.path.join(opt.output_path, prompts[i], "fake" + str(counter) +".png")
                        counter += 1
                        generated_images[i].save(dst_path)
