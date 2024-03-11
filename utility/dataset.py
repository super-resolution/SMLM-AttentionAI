import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from tifffile.tifffile import imread

import matplotlib.pyplot as plt
class CustomImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None, three_ch=False, offset=0):

        bg = imread("data/" + dataset + "/bg_images.tif")
        gt = np.load("data/" + dataset + "/ground_truth.npy", allow_pickle=True)
        idx = np.load("data/" + dataset + "/offsets.npy", allow_pickle=True)[offset:]
        images = imread("data/" + dataset + "/images.tif")[offset:].astype(np.float32)
        # for i in range(n_batches):
        point_length = idx[1:] - idx[:-1]
        truth = []
        mask = []
        self.batch_size = 100
        for batch in range(idx.shape[0]//self.batch_size):
            max_length = point_length[batch*self.batch_size:(batch+1)*self.batch_size].max()
            cur_truth = np.zeros((self.batch_size, max_length, 4))
            cur_mask = np.zeros((self.batch_size, max_length), dtype=np.float32)
            for i in range(self.batch_size):
                k = i+batch*self.batch_size
                val = gt[idx[k]:idx[k + 1]][:, (1, 0, 2, 3)] if gt.shape[1]==4 else gt[idx[k]:idx[k + 1]][:, (1, 0, 2)]
                cur_truth[i, :val.shape[0]] = val
                cur_mask[i, :val.shape[0]] = 1
            truth.append(cur_truth)
            mask.append(cur_mask)
        # truth = np.zeros((gt.shape[0], max_length, 4), dtype=np.float32)
        # mask = np.zeros((gt.shape[0], max_length), dtype=np.float32)
        # for i in range(idx.shape[0]-1):
        #     val = gt[idx[i]:idx[i + 1]][:, (1, 0, 2)]
        #     # if i%100 == 0:
        #     #     plt.imshow(images[i])
        #     #     plt.scatter(val[:,1],val[:,0])
        #     #     plt.show()
        #     truth[i, :val.shape[0], 0:2] = val[:, 0:2]
        #     truth[i, :val.shape[0], 2] = val[:, 2]
        #     mask[i, :val.shape[0]] = 1
        #
        #     if gt.shape[-1] == 4:
        #         truth[i, :val.shape[0], 3] = gt[idx[i]:idx[i + 1]][:, 3]

        self.truth = truth
        self.mask = mask
        if three_ch:

            self.images = self.reshape_data(images)
        else:
            self.images = images[:,None]
        self.transform = transform
        self.target_transform = target_transform
        self.bg = bg

    @staticmethod
    def reshape_data(images):
        #add temporal context to additional dimnesion
        dataset = np.zeros((images.shape[0],3,images.shape[1],images.shape[2]),dtype=np.int16)
        dataset[1:,0,:,:] = images[:-1]
        dataset[:,1,:,:] = images
        dataset[:-1,2,:,:] = images[1:]
        return dataset

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        batch = idx//self.batch_size
        i = idx%self.batch_size
        image = self.images[idx]
        truth = self.truth[batch][i]
        mask = self.mask[batch][i]
        bg = self.bg[truth[0,3].astype(np.int32)]
        return image, truth, mask, bg