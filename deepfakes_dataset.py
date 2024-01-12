import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np
import uuid
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ISONoise, MultiplicativeNoise, Cutout, CoarseDropout, MedianBlur, Blur, GlassBlur, MotionBlur, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, ToSepia, RandomShadow, RandomGamma, Rotate, Resize, RandomContrast, RandomBrightness, RandomBrightnessContrast

from transform.albu import IsotropicResize, CustomRandomCrop, FrequencyPatterns
import random

class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, size, pre_load_images = False, use_fake_patterns=False, pollute_pristines = False, mode = 'train'):
        self.x = images
        self.y = labels
        self.image_size = size
        self.pre_load_images = pre_load_images
        self.use_fake_patterns = use_fake_patterns
        self.pollute_pristines = pollute_pristines
        self.mode = mode
        self.n_samples = len(images)    
        
    def create_train_transforms(self, size = 224):
        return Compose([
                ImageCompression(quality_lower=40, quality_upper=100, p=0.2),
                GaussNoise(p=0.3),
                ISONoise(p=0.3),
                MultiplicativeNoise(p=0.3),
                HorizontalFlip(),
                OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                    CustomRandomCrop(size=size)
                ], p=1),
                Resize(height=size, width=size),
                PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                OneOf([RandomBrightnessContrast(), RandomContrast(), RandomBrightness(), FancyPCA(), HueSaturationValue()], p=0.5),
                OneOf([GaussianBlur(blur_limit=3), MedianBlur(), GlassBlur(), MotionBlur(), Blur()], p=0.5),
                OneOf([Cutout(), CoarseDropout()], p=0.05),
                ToGray(p=0.1),
                ToSepia(p=0.05),
                RandomShadow(p=0.05),
                RandomGamma(p=0.1),
                ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            ]
            )

    def create_train_fake_transform(self):
        return Compose([
            FrequencyPatterns(p=1)
        ])

    def create_val_transform(self, size):
        return Compose([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ])

    def __getitem__(self, index):
        if self.pre_load_images:
            image = self.x[index]
        else:
            image = cv2.imread(self.x[index])

        label = self.y[index]
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
            fake_transform = self.create_train_fake_transform()
        elif self.mode == "validation":
            transform = self.create_val_transform(self.image_size)

        
        if self.mode != "test":
            image = transform(image=image)['image']
            if  self.use_fake_patterns and self.mode == 'train':
                if label == 1:
                    image = fake_transform(image=image)['image']
                else:
                    if self.pollute_pristines and random.random() <= 0.10:
                        image = fake_transform(image=image)['image']
                        label = 1

            return torch.tensor(image).float(), float(label)
        else:
            images = transform(image=image)['image']
            return torch.tensor(images).float(), self.x[index], float(label)




    def __len__(self):
        return self.n_samples

 