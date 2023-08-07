import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from tifffile.tifffile import imread

class CustomImageDataset(Dataset):
    def __init__(self, dataset_cfg, transform=None, target_transform=None, train=True):
        if train:
            arr = np.load("data/" + dataset_cfg.train + "/coords.npy", allow_pickle=True)[:,::-1]
            indices = np.load("data/" + dataset_cfg.train + "/indices.npy", allow_pickle=True)[dataset_cfg.offset:]
            images = imread("data/" + dataset_cfg.train + "/images.tif")[dataset_cfg.offset:].astype(np.float32)

        else:
            arr = np.load("data/" + dataset_cfg.validation + "/coords.npy", allow_pickle=True)[:,::-1]
            indices = np.load("data/" + dataset_cfg.validation + "/indices.npy", allow_pickle=True)[dataset_cfg.offset:]
            images = imread("data/" + dataset_cfg.validation + "/images.tif")[dataset_cfg.offset:].astype(np.float32)

        max_length = max(np.sum(indices[:, :, 0], axis=1))
        truth = np.zeros((indices.shape[0], max_length, 2), dtype=np.float32)
        mask = np.zeros((indices.shape[0], max_length), dtype=np.float32)
        for i, ind in enumerate(indices):
            val = arr[np.where(ind[:, 0] != 0)]
            truth[i, :val.shape[0]] = val
            mask[i, :val.shape[0]] = 1

        self.truth = truth
        self.mask = mask
        self.images = self.reshape_data(images)
        self.transform = transform
        self.target_transform = target_transform

    @staticmethod
    def reshape_data(images):
        #add temporal context to additional dimnesion
        dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]))
        dataset[1:,0,:,:] = images[:-1]
        dataset[:,1,:,:] = images
        dataset[:-1,2,:,:] = images[1:]
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        truth = self.truth[idx]
        mask = self.mask[idx]
        return image, truth, mask