import os
import subprocess
import matplotlib.pyplot as plt
import re
os.environ["HYDRA_FULL_ERROR"] = "1"
def batch_training():
    networks = ["Decode","UNet","AttentionUNet","AttentionUNetdeep"]
    for net in networks:
        subprocess.call(["python", "train.py", f"training.base={net}", f"training.name={net}", f"network={net}", "hydra.job.chdir=False"])

def batch_training_hiddend():
    net = "DiffusionV3"
    hidden_d = [64,128]
    for d in hidden_d:
        subprocess.call(["python", "train.py", f"training.base={net}_{d}", f"training.name={net}_{d}",f"network.components.hidden_d={d}", f"network={net}", "hydra.job.chdir=False"])
def batch_fine_tune_hp():
    #todo: maybe train from scratch?
    pass


if __name__ == '__main__':
    batch_training_hiddend()
