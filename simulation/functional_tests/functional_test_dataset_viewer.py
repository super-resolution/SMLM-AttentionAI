import matplotlib.pyplot as plt

#todo: decrease triplet state
from utility.dataset import CustomTrianingDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataloader import default_collate

if __name__ == '__main__':
    datasets = CustomTrianingDataset("lab_logo_test", offset=0, )
    train_dataloader = DataLoader(datasets, batch_size=50, collate_fn=lambda x: tuple(x_.type(torch.float32) for x_ in default_collate(x)), shuffle=False)
    for images, truth, mask, bg in train_dataloader:
        for i in range(len(images)):
            plt.imshow(images[i,0])
            plt.scatter(truth[i,:,1],truth[i,:,0])
            plt.show()
