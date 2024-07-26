import os
import subprocess
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageOps, ImageDraw,ImageFont

def batch_evaluation_with_hydra():
    networks = ["AttentionUNetV2", "Decode"]
    files = [f"airyscan_approaching_lines"]
    for net in networks:
        #for ftune in ["","contest"]:
        subprocess.call(["python", "eval.py", f"dataset.name={','.join(files)}", f"training.name={net}", f"network={net}", "hydra.job.chdir=False"])


if __name__ == '__main__':
    batch_evaluation_with_hydra()