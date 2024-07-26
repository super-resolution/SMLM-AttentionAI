import os

import matplotlib.pyplot as plt
import torch
import numpy as np

if __name__ == '__main__':
    """
    plot training loss if multiple networks
    """
    # todo: get all trainings and compare loss
    models = os.listdir("trainings_compare")
    results = []
    fig,axs = plt.subplots(2,2)
    #map training name to a model feature
    #models = ["AttentionUNetV2","Decode","UNet"]

    models = {i:i for i in models
              }
    start = 0
    stop = 9
    # plot losses
    for model,name in models.items():
        if  "32" in name or "Decode"  in name:
            model_path = 'trainings_compare/{}'.format(model)
            checkpoint = torch.load(model_path)
            results = checkpoint["loss"]#if not "mini" in model else sum(c[1])
            #todo: norm on frame is currently 50
            #print(name,np.around(checkpoint["loss"][stop][1]/50,1))
            axs[0][0].plot([sum(c[1])  for c in checkpoint["loss"][start:stop]],label=name)
            axs[1][0].plot([c[1][0] for c in checkpoint["loss"][start:stop]],label=name)
            axs[0][1].plot([c[1][1] for c in checkpoint["loss"][start:stop]],label=name)
            axs[1][1].plot([c[1][2] for c in checkpoint["loss"][start:stop]],label=name)
    axs[0][0].set_title("tot loss")
    axs[1][0].set_title("pos loss")
    axs[0][1].set_title("count loss")
    axs[1][1].set_title("bg loss")
    plt.tight_layout()
    plt.legend()
    #save figure as svg in figures folder
    plt.savefig("figures/net_comparison.svg")
    plt.show()
