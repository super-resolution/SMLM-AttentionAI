import os
import subprocess
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageOps, ImageDraw,ImageFont

def batch_evaluation_with_hydra():
    #todo: flip before plot
    networks = ["DiffusionV3"]

    files = [f"lab_logo_dense{i}" for i in range(1,5)]
    for net in networks:
        subprocess.call(["python", "eval.py","-m", f"dataset.name={','.join(files)}", f"training.name={net}_64", f"network={net}", f"network.components.hidden_d={64}", f"dataset.save=density", "hydra.job.chdir=False"])

def plot_current_density_performances():
    files = os.listdir("figures/density")
    networks = ["Decode","DiffusionV3_64"]
    eval = "lab_logo_dense"
    rmse = []
    ji = []
    to_plot = [f for f in files if any([n in f for n in networks]) and eval in f and ".txt" in f]
    cmap = ["b","g","r","c","w"]
    i = 0
    for density_f in to_plot:
        with open("figures/density" + "\\" + density_f, "r") as f:
            s = f.readlines()[-2:]
            if "RMSE" in s[0]:
                RMSE = float(re.findall(r"[-+]?\d*\.?\d+",s[0] )[0])
            else:
                raise ValueError(f"No RMSE in {s[0]} check")
            if "JI" in s[1]:
                JI = float(re.findall(r"[-+]?\d*\.?\d+",s[1] )[0])
            else:
                raise ValueError(f"No JI in {s[1]} check")
            name = density_f.replace(eval,"").split(".")[0]
            density = name[0]
            print(f"{density} emitters/µm^2, {name[1:]} achieved a JI of {JI} and an RMSE of {RMSE}")
            marker = "X" if "decode" in name.lower() else "x"
            plt.scatter(JI, RMSE, c=cmap[i], marker=marker, label=fr"{density} emitter/$µm^2$ {name[1:]}")
            i += 1 if "diffusion" in name.lower() else 0
    plt.xlabel("JI")
    plt.ylabel("RMSE [nm]")
    plt.legend()
    plt.savefig("figures/density_comparison_scatter_plot.jpg")
    plt.show()

def make_figure():
    #todo: plot 4x2 60x40
    width,height = 4*600,2*400
    base_p = "figures/density/_final/"
    listofimages = os.listdir(base_p)
    cols = 4
    rows = 2
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width-1, thumbnail_height-1
    new_im = Image.new('RGB', (width, height))
    ims = []
    for i,p in enumerate(listofimages):
        i = i//2 if i%2==0 else i//2+rows+1
        im = Image.new("RGB", size)
        im.paste(Image.open(base_p+ p))
        I1 = ImageDraw.Draw(im)
        I1.text((10, 00),  chr(i+97), font=ImageFont.truetype("arial.ttf",86), fill=(255, 255, 255))
        im = ImageOps.expand(im,border=1,fill='white')
        #im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0
    new_im.save("Denisty.jpg")


if __name__ == '__main__':
    #batch_evaluation_with_hydra()
    plot_current_density_performances()
    #make_figure()