import os

import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    """
    plot training loss if multiple networks
    """
    # todo: get all trainings and compare loss
    models = os.listdir("trainings")
    results = []
    fig,axs = plt.subplots(2,2)
    #map training name to a model feature
    models = {"ViTVtest":"UNet",
              #"VITV3nlog":"",
              #"ViTV4":"UNet+Attention",
              "ViTV5miniV2":"UNet+Attention+UNet",
              #"ViTV5nlog":"a",
              #"ViTV5highpowerfinetune":"flim",
              #"ViTV6":"UNet+Attention+MLP+UNet",
              "ViTV7miniV2":"UNetdeep+Attention+UNetdeep",
              "ViTV8mini": "UNetdeep+PosEncoding+Attention+UNetdeep",
              "DecodeV4":"Decode",
              "DecodeV3": "Decode1ch"

              }
    start = 0
    stop = 9999
    # plot losses
    for model,name in models.items():
        model_path = 'trainings/model_{}'.format(model)
        checkpoint = torch.load(model_path)
        results = checkpoint["loss"]#if not "mini" in model else sum(c[1])
        axs[0][0].plot([sum(c[1])  for c in checkpoint["loss"][start:stop]],label=name)
        axs[1][0].plot([c[1][0] for c in checkpoint["loss"][start:stop]],label=name)
        axs[0][1].plot([c[1][1] for c in checkpoint["loss"][start:stop]],label=name)
        axs[1][1].plot([c[1][2] for c in checkpoint["loss"][start:stop]],label=name)
    plt.legend()
    #save figure as svg in figures folder
    plt.savefig("figures/net_comparison.svg")
    plt.show()
