from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from ldm.data.image_folder import make_dataset
from torch.utils.data.distributed import DistributedSampler

class GestaltDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train=None, validation=None,
                 test=None, num_workers=4, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        dataset = GestaltData(root_dir = self.root_dir, validation=False)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler, drop_last=True)

    def val_dataloader(self):
        dataset = GestaltData(root_dir = self.root_dir, validation=True)
        sampler = DistributedSampler(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class GestaltData(Dataset):
    def __init__(self,
            root_dir,
            validation=False, 
            is_trainsize_ablation = False,
            ) -> None:
        self.root_dir = Path(root_dir)

        self.occlusion_root = os.path.join(root_dir, 'occlusion')
        self.whole_root = os.path.join(root_dir, 'whole')
        self.whole_mask_root = os.path.join(root_dir, 'whole_mask')
        self.visible_mask_root = os.path.join(root_dir, 'visible_object_mask')

        all_occlusion_paths = sorted(make_dataset(self.occlusion_root))
        all_whole_paths = sorted(make_dataset(self.whole_root))
        all_whole_mask_paths = sorted(make_dataset(self.whole_mask_root))
        all_visible_mask_paths = sorted(make_dataset(self.visible_mask_root))

        total_objects = len(all_occlusion_paths)
        print("Total number of samples: %d" % total_objects)
        val_start = math.floor(total_objects / 100. * 99.)
        if validation:
            # Last 1% as validation
            self.occlusion_paths = all_occlusion_paths[val_start:]
            self.whole_paths = all_whole_paths[val_start:]
            self.whole_mask_paths = all_whole_mask_paths[val_start:]
            self.visible_mask_paths = all_visible_mask_paths[val_start:]
        else:
            # First 99% as training
            self.occlusion_paths = all_occlusion_paths[:val_start]
            self.whole_paths = all_whole_paths[:val_start]
            self.whole_mask_paths = all_whole_mask_paths[:val_start]
            self.visible_mask_paths = all_visible_mask_paths[:val_start]
        print('============= length of %s dataset %d =============' % ("validation" if validation else "training", self.__len__()))

    def __len__(self):
        return len(self.occlusion_paths)

    def __getitem__(self, index):
        data = {}
    
        visible_mask_path = self.visible_mask_paths[index]
        occlusion_image_path =  self.occlusion_paths[index]
        whole_image_path = self.whole_paths[index]

        occluded_object_image = read_rgb(occlusion_image_path)
        whole_object_image = read_rgb(whole_image_path)
        visible_mask = read_mask(visible_mask_path)

        rgb_visible_mask = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
        rgb_visible_mask[:,:,0] = visible_mask
        rgb_visible_mask[:,:,1] = visible_mask
        rgb_visible_mask[:,:,2] = visible_mask

        data["image_cond"] = self.process_image(occluded_object_image) # input occlusion image
        data["visible_mask_cond"] = self.process_image(rgb_visible_mask) # input visible (modal) mask
        data["image_target"] = self.process_image(whole_object_image) # target whole (amodal) image
        return data

    def process_image(self, input_im):
        input_im = input_im.astype(float) / 255. # [0, 255] to [0., 1.]
        normalized_image = input_im * 2 - 1 # [0, 1] to [-1, 1]
        return normalized_image

def read_mask(file_path):
    """
    In:
        file_path: Path to binary mask png image.
    Out:
        binary mask as np array [height, width].
    Purpose:
        Read in a mask image.
    """
    return cv2.imread(file_path, -1)

def read_rgb(file_path):
    """
    In:
        file_path: Color image png to read.
    Out:
        RGB image as np array [height, width, 3], each value in range [0, 255]. Color channel in the order RGB.
    Purpose:
        Read in a color image.
    """
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)