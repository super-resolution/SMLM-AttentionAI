import os

import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    # todo: get all trainings and compare loss
    models = os.listdir("trainings")
    results = []
    fig,axs = plt.subplots(2,2)
    models = {"ViTV3":"UNet",
              "VITV3nlog":"",
              "ViTV4":"UNet+Attention",
              "ViTV5":"UNet+Attention+UNet",
              "ViTV5nlog":"a",
              "ViTV5highpowerfinetune":"flim",
              "ViTV6":"UNet+Attention+MLP+UNet"
    }
    start = 0
    stop = 9999
    for model,name in models.items():
        model_path = 'trainings/model_{}'.format(model)
        checkpoint = torch.load(model_path)
        results = checkpoint["loss"]
        axs[0][0].plot([sum(c[1]) for c in checkpoint["loss"][start:stop]],label=name)
        axs[1][0].plot([c[1][0] for c in checkpoint["loss"][start:stop]],label=name)
        axs[0][1].plot([c[1][1] for c in checkpoint["loss"][start:stop]],label=name)
        axs[1][1].plot([c[1][2] for c in checkpoint["loss"][start:stop]],label=name)
    plt.legend()
    plt.savefig("figures/net_comparison.svg")
    plt.show()
