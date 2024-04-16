import numpy as np
import torch

class DropoutBox():
    def __init__(self,m, device="cuda"):
        self.device = device
        self.max = m
        box = np.random.rand(2)
        x_min = box.min()
        x_max = box.max()
        box = np.random.rand(2)
        y_min = box.min()
        y_max = box.max()
        self.box = {"x_min":x_min,"x_max":x_max,"y_min":y_min,"y_max":y_max}
        for k,v in self.box.items():
            self.box[k] = torch.tensor([v], device=self.device)

    def update_box(self):
        box = np.random.rand(2)
        self.box["x_min"] = torch.tensor([box.min()], device=self.device)
        self.box["x_max"] = torch.tensor([box.max()], device=self.device)
        box = np.random.rand(2)
        self.box["y_min"] = torch.tensor([box.min()], device=self.device)
        self.box["y_max"] = torch.tensor([box.max()], device=self.device)

    def forward(self, arr, indices):
        if arr.shape[0]==0:
            return []
        dropout_indices_relative = torch.where((arr[indices,0] < self.box["x_min"]* self.max)|
                              (arr[indices,0] > self.box["x_max"]* self.max)|
                              (arr[indices,1] > self.box["y_max"]* self.max)|
                              (arr[indices,1] < self.box["y_min"]* self.max))
        #todo: test this
        indices = indices[dropout_indices_relative]
        # data augmentation discard data within box
        return indices, dropout_indices_relative