import os
import math
import random
import numpy as np
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.transform import *


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.reduce_zero_label = True if name == 'ade20k' else False
        
        if mode == "train_l":
            self.root = os.path.join(root, "train")
            self.ids = os.listdir(os.path.join(self.root,"target"))
        elif mode == "train_u":
            self.root = os.path.join(root, "train")
            self.ids = list(set(os.listdir(os.path.join(self.root, "input"))) - set(os.listdir(os.path.join(self.root, "target"))))
        elif mode == "test":
            self.root = os.path.join(root, "test")
            self.ids = os.listdir(os.path.join(self.root, "target"))
        else:
            raise ValueError("%s is not available mode", mode)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, "input", id)).convert('RGB')
        if (self.mode == "train_l") or (self.mode == "test") or (self.mode == "val"):
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, "target", id))))
        else:
            mask = Image.fromarray(np.zeros(img.size))

        # if self.reduce_zero_label:
        #     mask = np.array(mask)
        #     mask[mask == 0] = 255
        #     mask = mask - 1
        #     mask[mask == 254] = 255
        #     mask = Image.fromarray(mask)

        if self.mode == 'val' or self.mode == "test":
            img_ori = transforms.functional.to_tensor(img)
            img, mask = normalize(img, mask)
            return img, mask, id, img_ori

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)

# class SemiDataset(Dataset):
#     def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
#         self.name = name
#         self.root = root
#         self.mode = mode
#         self.size = size
#         self.reduce_zero_label = True if name == 'ade20k' else False

#         if mode == 'train_l' or mode == 'train_u':
#             with open(id_path, 'r') as f:
#                 self.ids = f.read().splitlines()
#             if mode == 'train_l' and nsample is not None:
#                 self.ids *= math.ceil(nsample / len(self.ids))
#                 self.ids = self.ids[:nsample]
#         else:
#             with open('splits/%s/val.txt' % name, 'r') as f:
#                 self.ids = f.read().splitlines()

#     def __getitem__(self, item):
#         id = self.ids[item]
#         img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
#         mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

#         if self.reduce_zero_label:
#             mask = np.array(mask)
#             mask[mask == 0] = 255
#             mask = mask - 1
#             mask[mask == 254] = 255
#             mask = Image.fromarray(mask)

#         if self.mode == 'val' or self.mode == "test":
#             img, mask = normalize(img, mask)
#             return img, mask, id

#         img, mask = resize(img, mask, (0.5, 2.0))
#         ignore_value = 254 if self.mode == 'train_u' else 255
#         img, mask = crop(img, mask, self.size, ignore_value)
#         img, mask = hflip(img, mask, p=0.5)

#         if self.mode == 'train_l':
#             return normalize(img, mask)

#         img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

#         if random.random() < 0.8:
#             img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
#         img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
#         img_s1 = blur(img_s1, p=0.5)
#         cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

#         if random.random() < 0.8:
#             img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
#         img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
#         img_s2 = blur(img_s2, p=0.5)
#         cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

#         ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

#         img_s1, ignore_mask = normalize(img_s1, ignore_mask)
#         img_s2 = normalize(img_s2)

#         mask = torch.from_numpy(np.array(mask)).long()
#         ignore_mask[mask == 254] = 255

#         return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

#     def __len__(self):
#         return len(self.ids)
