import os
import subprocess
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageOps, ImageDraw,ImageFont

def batch_fine_tune_ld():
    ld_dataset = "airyscan_ld"
    networks = ["AttentionUNetV2",]
    name = "_ld"
    for net in networks:
        subprocess.call(["python", "train.py",f"dataset.validation=crlb_test2", f"dataset.train=[{ld_dataset}]", f"training.base={None}", f"training.name={net+name}", f"network={net}", "hydra.job.chdir=False"])

def batch_evaluation_with_hydra():
    networks = ["AttentionUNetV2","Decode"]
    files = [f"crlb_test"]
    for net in networks:
        for ftune in ["_ld"]:
            subprocess.call(["python", "eval.py", f"dataset.name={','.join(files)}", f"training.name={net+ftune}", f"network={net}", "hydra.job.chdir=False"])

if __name__ == '__main__':
    batch_evaluation_with_hydra()