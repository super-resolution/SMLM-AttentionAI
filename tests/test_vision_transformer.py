from unittest import TestCase

import matplotlib.pyplot as plt
import torch
from tifffile.tifffile import imread
import torch.nn as nn
import numpy as np
from models.VIT import ImageEmbedding,ViT,Encoder

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result





class TestVisionTransformer():
    def __init__(self) -> None:
        self.images = torch.Tensor(imread("data/" + "lab_logo" + "/images.tif")[500:600, 0:60, 0:60])

    def test_patchify(self):
        fig, axs = plt.subplots(4)
        axs[0].imshow(self.images[0])
        T = ImageEmbedding()
        patches = T.patchify(self.images)
        print(patches.shape)
        axs[1].imshow(patches[0,0,0])
        axs[2].imshow(patches[0,0,1])
        #todo: test flatten function
        patches_flat = patches.flatten(start_dim=1)
        #patch flat works
        axs[3].imshow(torch.reshape(patches_flat[0,81:2*81],(9,-1)))
        plt.show()

    def test_embedding(self):
        T = ImageEmbedding(hidden_d=1600)
        result = T.forward(self.images)
        image_space = result.unfold(1,6,6).unfold(2,40,40)
        #image_space = torch.permute(image_space, (0,1,3,2,4))
        image_space = torch.reshape(image_space,(-1,image_space.shape[1]*image_space.shape[2], image_space.shape[3]*image_space.shape[4]))
        plt.imshow(image_space[1,:,:].detach().numpy())
        plt.show()
        print(result.shape)

    def test_encoder(self):
        T = Encoder()
        #encoder working as expected
        result = T.forward(self.images)
        print(result.shape)

    def test_vit(self):
        T = ViT()
        #encoder working as expected
        result = T.forward(self.images)
        print(result.shape)
        fig,axs = plt.subplots(3)
        for i in range(3):
            axs[i].imshow(result[0,i].detach().numpy())
        plt.show()

    def test_fold_unfold(self):
        plt.imshow(self.images[0])
        plt.show()
        #test folding and unfolding works as expected
        patches = self.images.unfold(1, 10, 10).unfold(2, 10, 10)
        image_space = torch.reshape(patches, (patches.shape[0], patches.shape[1] * patches.shape[2], -1))
        image_space = image_space.repeat(1,1,8)
        image_space = image_space[:,:36,:].unfold(2,100,100)
        image_space = torch.permute(image_space, dims=(0,2,1,3))
        image_space = image_space[:,:,:,:].unfold(2,6,6).unfold(3,10,10)
        image_space = torch.reshape(image_space,(-1,8, image_space.shape[2]*image_space.shape[3], image_space.shape[4]*image_space.shape[5]))
        plt.imshow(image_space[0,4,:,:])
        plt.show()

if __name__ == '__main__':
    T = TestVisionTransformer()
    T.test_vit()
