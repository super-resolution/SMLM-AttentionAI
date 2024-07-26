import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def plot_curves_to_one_plot(fp):
    df = pd.read_csv(fp)
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    df = df.map(lambda s: np.array([float(v.strip("[]")) for v in s.split(",")]))
    fig,axs = plt.subplots(4,4,figsize=[16,10])
    densities = 4
    parameters = ["sig","p"]
    networks = ["Decode","AttentionUNetV2"]
    base = "lab_logo_dense"
    for dens in range(densities):
        density = base+str(dens+1)
        axs[dens][0].annotate(f"Density {dens+1}", xy=(0, 0.5), xytext=(-axs[dens][0].yaxis.labelpad - 5, 0),
                xycoords=axs[dens][0].yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
        for i,net in enumerate(networks):
            for j,param in enumerate(parameters):
                ax = axs[dens][2*i+j]
                if dens ==0:
                    ax.annotate(f"{net} {param}", xy=(0.5, 1), xytext=(0, 5),
                                xycoords='axes fraction', textcoords='offset points',
                                size='large', ha='center', va='baseline')
                col = density+net+param
                JI = np.stack(df[col+"_Jaccard"].values)
                RMSE = np.stack(df[col+"_RMSE"].values)
                #ax.set_title(col)
                ax.plot(JI[:,0],JI[:,1],'#378f8f', lw=3)
                ax.tick_params(axis='y', labelcolor="#378f8f")
                ax.set_ylabel('JI', color="#378f8f")
                ax.set_xlabel("threshold")
                ax2 = ax.twinx()
                ax2.plot(RMSE[:,0],RMSE[:,1],'#ff6150', lw=3)
                ax2.tick_params(axis='y', labelcolor='#ff6150')
                ax2.set_ylabel('rmse', color='#ff6150')
                #axs2.set_ylim([0,max(rmse[:,1])])
                fig.tight_layout()
    plt.show()


def plot_best_to_latex_table(fp):


    df = pd.read_csv(fp)
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    df = df.map(lambda s: np.array([float(v.strip("[]")) for v in s.split(",")]))
    #todo: select where JI is largest
    #todo: also plot loss?
    densities = 1
    parameters = ["p"]
    networks = ["Decode","AttentionUNetV2","Decodecontest","AttentionUNetV2contest"]
    base = "ContestHD"#"lab_logo_dense
    df2 = defaultdict(list)
    for dens in range(densities):
        density = base#+str(dens+1)
        for i,net in enumerate(networks):
            for j,param in enumerate(parameters):
                col = density+net+param
                JI = np.stack(df[col+"_Jaccard"].values)
                ind = np.argmax(JI[:,1])
                RMSE = np.stack(df[col+"_RMSE"].values)
                #todo: stack densities
                df2[net+" JI"].append(JI[ind,1])
                df2[net+" RMSE"].append(RMSE[ind,1])
    df2 = pd.DataFrame(df2)
    print(df2.to_latex(float_format="{:.2f}".format))

if __name__ == '__main__':
    plot_best_to_latex_table(r"C:\Users\biophys\PycharmProjects\pytorchDataSIM\figures\threshold_contest.csv")