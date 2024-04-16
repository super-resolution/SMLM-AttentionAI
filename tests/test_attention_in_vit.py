import unittest

import matplotlib.pyplot as plt
import torch
from tifffile.tifffile import imread
import numpy as np
from models.VIT.vitv5 import ViT
from hydra import initialize, compose
from utility.dataset import CustomTrianingDataset
from torch.utils.data import DataLoader


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class TestVisionDecoder(unittest.TestCase):
    def setUp(self) -> None:
        with initialize(version_base=None, config_path="../cfg/network"):
            cfg = compose(config_name="ViTV5", )
            self.net = ViT(cfg.components)
            checkpoint = torch.load("../trainings/model_ViTV5")
            state_dict = checkpoint['model_state_dict']
            self.net.load_state_dict(state_dict)
            #todo: use dataloader and find corresponding localisations
        datasets = CustomTrianingDataset(cfg.dataset.name, offset=cfg.dataset.offset)
        dataloader = DataLoader(datasets, batch_size=cfg.dataset.batch_size, collate_fn=lambda x: tuple(
            x_.type(torch.float32).to(device) for x_ in default_collate(x)), shuffle=True)

    def test_attention(self):
        res = self.net(self.images)