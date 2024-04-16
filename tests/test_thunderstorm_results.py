import importlib

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from utility.dataset import CustomTrianingDataset
from utility.emitters import Emitter
from visualization.visualization import plot_emitter_set




@hydra.main(config_name="eval.yaml", config_path="../cfg")
def myapp(cfg):
    device = cfg.network.device
    dataset_name = cfg.dataset.name
    dataset_offset = cfg.dataset.offset
    #todo: set three channel true if decode
    three_ch = "decode" in cfg.training.name.lower()
    datasets = CustomTrianingDataset(cfg.dataset.name, three_ch=three_ch, offset=cfg.dataset.offset)
    dataloader = DataLoader(datasets, batch_size=cfg.dataset.batch_size,collate_fn=lambda x: tuple(x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=False)
    thunderstorm = Emitter.from_thunderstorm_csv(r"C:\Users\biophys\PycharmProjects\pytorchDataSIM\data\\" + cfg.dataset.name + "/thunderstorm_results.csv")
    emitter_truth = None
    for images, truth, mask, bg in dataloader:
        if not emitter_truth:
            emitter_truth = Emitter.from_ground_truth(truth.cpu().numpy())
        else:
            emitter_truth + Emitter.from_ground_truth(truth.cpu().numpy())
    thunderstorm = thunderstorm.filter(photons=100)
    thunderstorm.compute_jaccard(emitter_truth, np.concatenate([im.cpu().numpy() for im,_,_,_ in dataloader],axis=0))
    plot_emitter_set(thunderstorm)

if __name__ == '__main__':
    myapp()
