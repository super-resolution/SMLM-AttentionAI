import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from tifffile.tifffile import imread

import matplotlib.pyplot as plt
class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, offset=0):

        bg = imread("data/" + dataset + "/bg_images.tif")
        gt = np.load("data/" + dataset + "/ground_truth.npy", allow_pickle=True)
        idx = np.load("data/" + dataset + "/offsets.npy", allow_pickle=True)[offset:]
        images = imread("data/" + dataset + "/images.tif")[offset:].astype(np.float32)
        max_length = max(idx[1:]-idx[:-1])
        truth = np.zeros((gt.shape[0], max_length, 4), dtype=np.float32)
        mask = np.zeros((gt.shape[0], max_length), dtype=np.float32)
        for i in range(idx.shape[0]-1):
            val = gt[idx[i]:idx[i + 1]][:, (1, 0, 2)]
            # if i%100 == 0:
            #     plt.imshow(images[i])
            #     plt.scatter(val[:,1],val[:,0])
            #     plt.show()
            truth[i, :val.shape[0], 0:2] = val[:, 0:2]
            truth[i, :val.shape[0], 2] = val[:, 2]
            mask[i, :val.shape[0]] = 1

            if gt.shape[-1] == 4:
                truth[i, :val.shape[0], 3] = gt[idx[i]:idx[i + 1]][:, 3]

        self.truth = truth
        self.mask = mask
        self.images = self.reshape_data(images)
        self.transform = transform
        self.target_transform = target_transform
        self.bg = bg

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
        bg = self.bg[truth[0,3].astype(np.int32)]
        return image, truth, mask, bg