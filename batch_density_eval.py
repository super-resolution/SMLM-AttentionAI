import os
import subprocess
import matplotlib.pyplot as plt
import re

def batch_evaluation_with_hydra():
    networks = ["AttentionUNetV2","Decode"]

    files = [f"lab_logo_dense{i}" for i in range(1,5)]
    for net in networks:
        subprocess.call(["python", "eval.py","-m", f"dataset.name={','.join(files)}", f"training.name={net}", f"network={net}", "hydra.job.chdir=False"])

def plot_current_density_performances():
    files = os.listdir("figures/density")
    networks = ["Decode","AttentionUNetV2"]
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
            i += 1 if "decode" in name.lower() else 0
    plt.xlabel("JI")
    plt.ylabel("RMSE [nm]")
    plt.legend()
    plt.savefig("figures/density_comparison_scatter_plot.jpg")
    plt.show()


if __name__ == '__main__':
    batch_evaluation_with_hydra()
    plot_current_density_performances()