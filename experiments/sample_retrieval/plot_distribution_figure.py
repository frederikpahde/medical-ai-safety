import torch
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns
from utils.plots import add_border
from matplotlib.patches import ConnectionPatch
from torchvision.utils import make_grid
from crp.image import imgify
from torchvision.transforms import Resize

def create_data_annotation_plot(data_pd, dataset_split, idxs_interesting_clean, idxs_interesting_art, 
                                localizations, plot_connections=True, savename=""):

    nrows, ncols = 2, 1
    s = 1.4
    resize = Resize((150, 150))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*s,5*s), 
                            gridspec_kw={'height_ratios': [5,5]})


    ax1 = axs[0]
    sns.kdeplot(data=data_pd, 
                x="value", 
                ax=ax1,
                hue="artifact_label", 
                legend=False,
                fill=True)

    ax1.set_xlabel("")

    w = 10
    col_clean = (torch.Tensor(sns.color_palette()[0]) * 255).type(torch.uint8)
    imgs_clean = [resize(add_border(dataset_split.reverse_normalization(dataset_split[idx][0]),
                            col_clean, w)) for idx in idxs_interesting_clean]

    col_art = (torch.Tensor(sns.color_palette()[1]) * 255).type(torch.uint8)
    imgs_art = [resize(add_border(dataset_split.reverse_normalization(dataset_split[idx][0]),
                            col_art, w)) for idx in idxs_interesting_art]
    
    localizations /= localizations.max()
    imgs_localizations = [
        resize(add_border(torch.Tensor(np.array(imgify(localizations[i], cmap="bwr", 
                                                       symmetric=True, level=3.0, 
                                                       vmin=-1, vmax=1).convert("RGB")).transpose(2,0,1)).type(torch.uint8),
                torch.Tensor([0,0,0]),w // 2))
        for i in range(len(localizations))]

    grid = make_grid(imgs_clean + imgs_art + imgs_localizations,
                     padding=10, pad_value=255, nrow=len(imgs_localizations))

    ax2 = axs[1]
    ax2.imshow(grid.numpy().transpose(1,2,0))
    ax2.axis("off")

    if plot_connections:
        width_ax2 = ax2.get_xlim()[1]
        pos_x2_connector = [width_ax2 / 12 * i for i in [1,3,5,7,9,11]]
        for i, value in enumerate(data_pd.loc[idxs_interesting_clean].value.values):
            ax1.scatter(x=value, y=0, color=sns.color_palette()[0], marker='o', s=50)
            con = ConnectionPatch(xyA=(value,0), 
                                xyB=(pos_x2_connector[i], 10), 
                                coordsA=ax1.transData, coordsB=ax2.transData, 
                                color=sns.color_palette()[0], alpha=.7)
            fig.add_artist(con)
            
        for i, value in enumerate(data_pd.loc[idxs_interesting_art].value.values):
            ax1.scatter(x=value, y=0, color=sns.color_palette()[1], marker='o', s=50)
            con = ConnectionPatch(xyA=(value,0), 
                                xyB=(pos_x2_connector[i+3], 10), 
                                coordsA=ax1.transData, coordsB=ax2.transData, 
                                color=sns.color_palette()[1], alpha=.7)
            fig.add_artist(con)

    ax1.set_xlabel("")
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("")
    ax1.set_title("")

    plt.tight_layout()
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    fig.savefig(savename, bbox_inches="tight", dpi=500)