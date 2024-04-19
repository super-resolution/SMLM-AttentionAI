import os
import subprocess
import matplotlib.pyplot as plt
import re
os.environ["HYDRA_FULL_ERROR"] = "1"
def batch_evaluation_with_hydra():
    networks = ["Decode","UNet","AttentionUNet","AttentionUNetdeep"]
    for net in networks:
        subprocess.call(["python", "train.py", f"training.base={net}", f"training.name={net}", f"network={net}", "hydra.job.chdir=False"])

if __name__ == '__main__':
    batch_evaluation_with_hydra()
