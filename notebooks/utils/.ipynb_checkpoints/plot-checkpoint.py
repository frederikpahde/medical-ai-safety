from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from crp.image import imgify, zimage
from torchvision.utils import make_grid

def plot_data(ds, start_idx=0, nrows=2, ncols=8, dpi=100):
    size = 2.5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False, dpi=dpi)
    for i in range(min(nrows * ncols, len(ds))):
        ax = axs[i // ncols][i % ncols]
        idx = start_idx + i
        batch = ds[idx]
        if len(batch) == 2:
            img, y = batch
        else:
            img, y, _ = batch
        img = np.moveaxis(ds.reverse_normalization(img).numpy(), 0, 2)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{ds.classes[y.item()]}")
        
def plot_local_explanations(samples, ys, preds, hms, classes, cmap="bwr", level=1, dpi=100):
    nrows = 2
    ncols = len(samples)
    size = 2.5

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False, dpi=dpi)
    for i, (x, y, pred, hm) in enumerate(zip(samples, ys, preds, hms)):
        img = np.moveaxis(x, 0, 2)
        axs[0, i].imshow(img)
        axs[0, i].axis("off")
        axs[0, i].set_title(f"Ground truth: {classes[y.item()]}\nPred.: {classes[pred.item()]}")


        axs[1, i].imshow(imgify(hm, vmin=-1, vmax=1, cmap="bwr", level=2))
        axs[1, i].axis("off")

def plot_local_explanations_corrected(samples, ys, preds, preds_corrected, hms, hms_corrected, classes, cmap="bwr", level=1, dpi=100):
    nrows = 3
    ncols = len(samples)
    size = 2.5

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(size * ncols, size * nrows), squeeze=False, dpi=dpi)
    for i, (x, y, pred, pred_corrected, hm, hm_corrected) in enumerate(zip(samples, ys, preds, preds_corrected, hms, hms_corrected)):
        img = np.moveaxis(x, 0, 2)
        axs[0, i].imshow(img)
        axs[0, i].axis("off")
        axs[0, i].set_title(f"Label: {classes[y.item()]}")


        axs[1, i].imshow(imgify(hm, vmin=-1, vmax=1, cmap="bwr", level=2))
        axs[1, i].set_title(f"Pred.: {classes[pred.item()]}")
        axs[1, i].axis("off")
        
        axs[2, i].imshow(imgify(hm_corrected, vmin=-1, vmax=1, cmap="bwr", level=2))
        axs[2, i].set_title(f"Pred.: {classes[pred_corrected.item()]}")
        axs[2, i].axis("off")
        
def remove_ticks(ax):
    ax.set_yticks([])
    ax.set_xticks([])

def plot_glocal_explanation(sample, attr, cond_heatmap, n_concepts, n_refimgs, 
                            topk_ind, topk_rel, ref_imgs, target, pred, y, classes, dpi=120):
    base_size = 2
    cols = 3
    width_ratios = [1] * (cols-1) + [n_refimgs / 4]
    fig, axs = plt.subplots(n_concepts, cols, gridspec_kw={'width_ratios': width_ratios},
                            figsize=(base_size * cols * 1.1, base_size * n_concepts), dpi=dpi, squeeze=False)
    level = 2
    resize = T.Resize((150, 150))
    
    # Input Image
    ax = axs[0][0]
    ax.set_title(f"input ({classes[pred]}/gt: {classes[y]})")
    ax.imshow(sample.numpy().transpose(1,2,0))
    
    # Heatmap (local explanation)
    ax = axs[1][0]
    ax.set_title("heatmap")
    amax = attr.heatmap.abs().max()
    img = imgify(attr.heatmap / amax, cmap="bwr", vmin=-1, vmax=1, symmetric=True, level=level)
    ax.imshow(img)
    
    [_axs[0].axis("off") for _axs in axs[2:]]
    
    for i in range(n_concepts):
        
        # Concept Heatmap
        ax = axs[i][1]
        if i == 0:
            ax.set_title("cond. heatmap")
        ax.imshow(imgify(cond_heatmap[i], symmetric=True, cmap="bwr", padding=True))
        ax.set_ylabel(f"concept {topk_ind[i]}\n relevance: {(topk_rel[i] * 100):2.1f}%")
        
        # Concept Visualization with Reference Samples
        ax = axs[i][2]
        if i == 0:
            ax.set_title("concept")
        grid = make_grid(
            [resize(torch.from_numpy(np.copy(np.asarray(img))).permute((2, 0, 1))) for img in ref_imgs[topk_ind[i]]],
            nrow=int(n_refimgs / 2),
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.yaxis.set_label_position("right")


    [remove_ticks(ax) for _axs in axs for ax in _axs]
    plt.tight_layout()
    plt.show()

def show_relmax_refimgs(ref_imgs, axs):
    resize = T.Resize((150, 150))
    for r, (concept, imgs) in enumerate(ref_imgs.items()):
        ax = axs[r][0]
        grid = make_grid(
        [resize(torch.from_numpy(np.copy(np.asarray(img))).permute((2, 0, 1))) for img in imgs],
            padding=2)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"Concept {concept}")
        
def plot_2d(data, label, ax, axis_labels={"x": "Dim 1", "y": "Dim 2"}):
    data = pd.DataFrame(data)
    data.columns = ['x', 'y']
    data['label'] = label
    
    palette =sns.color_palette()[:len(np.unique(data['label'].values))]
    
    # others
    sns.scatterplot(data=data[data['label'] == 0], x="x", y="y", 
                    color="grey", size=5, alpha=.4, ax=ax, legend=False)

    # highlighted
    sns.scatterplot(
        data=data[data['label'] != 0].sort_values("label"), x="x", y="y", 
        hue = 'label', 
        palette = palette[1:],
        s=150, alpha = 0.8, legend = False,
        ax = ax
    )

    ax.set(xlabel=axis_labels["x"], ylabel=axis_labels["y"])
    sns.despine()

def show_outlier_imgs(imgs, ax):
    grid = make_grid([img for img in imgs], padding=2, nrow=6)
    ax.imshow(grid.numpy().transpose(1,2,0))
    ax.set_yticks([]); ax.set_xticks([])
    
def plot_2d_data_embedding(data_2d, outlier_labels, outlier_imgs, dpi=80):
    nrows = 3
    ncols = 1
    base_size = 1.8
    mul_emb = 5
    gap = .1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(base_size * mul_emb, base_size * mul_emb * 1.5), 
                            gridspec_kw={'height_ratios':[mul_emb] + [gap] + [mul_emb / 2]}, dpi=dpi)
    
    
    ax_emb = axs[0]
    plot_2d(data_2d, outlier_labels, ax_emb)
    axs[1].axis("off"); 
    ax_emb.axis("off")
    
    show_outlier_imgs(outlier_imgs, axs[2])
    
    fig.subplots_adjust(hspace=.2)
    fig.show()
    
def plot_2d_concept_embedding(data_2d, outlier_concepts, ref_imgs_outlier_concepts, dpi=60):
    nconcepts = len(outlier_concepts)
    nrows = 1 + nconcepts + 1
    ncols = 1
    base_size = 1.8
    mul_emb = 5
    gap = .2


    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size * mul_emb, (nrows-3) * base_size + base_size * mul_emb), 
                            gridspec_kw={'height_ratios':[mul_emb] + [gap] + [1] * nconcepts}, dpi=dpi, squeeze=False)

    
    concept_labels = np.zeros(len(data_2d)).astype(np.uint8)
    concept_labels[outlier_concepts] = 1
    
    ax_emb = axs[0][0]
    plot_2d(data_2d, concept_labels, ax_emb)
    axs[1][0].axis("off"); 

    ref_imgs_reduced = {c: ref_imgs_outlier_concepts[c] for c in outlier_concepts}
    axs_relmax = axs[2:2+nconcepts]

    show_relmax_refimgs(ref_imgs_reduced, axs_relmax)
    if len(axs_relmax) > 0:
        axs_relmax[0][0].set_title("RelMax")

    ax_emb.axis("off")
    fig.subplots_adjust(hspace=.2)
    fig.show()
    


def plot_pcx_matrix(concept_matrix, dataset, prototype_samples, ref_imgs,
                counts, counts_perc, topk_ind, num_ref_concept, top_k_sample_prototype, num_hms, savename):
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    n_concepts, n_prototypes = concept_matrix.shape

    fig, axs = plt.subplots(nrows=n_concepts + 1, ncols=n_prototypes + 1, figsize=(n_prototypes + num_ref_concept, 
                                                                                   n_concepts + top_k_sample_prototype), dpi=100,
                            gridspec_kw={'width_ratios': [num_ref_concept] + [1 for _ in range(n_prototypes)],
                                         'height_ratios': [top_k_sample_prototype] + [1 for _ in range(n_concepts)]})
    
    
    for i in range(n_concepts):
        for j in range(n_prototypes):
            val = concept_matrix[i, j].item()
            axs[i + 1, j + 1].matshow(np.ones((1, 1)) * val if val >= 0 else np.ones((1, 1)) * val * -1,
                                      vmin=0,
                                      vmax=concept_matrix.abs().max(),
                                      cmap="Reds" if val >= 0 else "Blues")
            minmax = concept_matrix.abs().max() * 100 / 2
            cos = val * 100
            color = "white" if abs(cos) > minmax else "black"
            axs[i + 1, j + 1].text(0, 0, f"{cos:.1f}", ha="center", va="center", color=color, fontsize=15)
            axs[i + 1, j + 1].axis('off')

    resize = T.Resize((120, 120))
    for i in range(n_prototypes):
        grid = make_grid(
            [resize(dataset[prototype_samples[j][i]][0])
             for j in range(top_k_sample_prototype)],
            nrow=1,
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        img = imgify(grid)
        axs[0, i + 1].imshow(img)
        axs[0, i + 1].set_xticks([])
        axs[0, i + 1].set_yticks([])
        axs[0, i + 1].set_title(f"prototype {i} \ncovers {counts[i]} \n({counts_perc[i]:.0f}\%)")
        axs[0, 0].axis('off')


    for i in range(n_concepts):
        grid = make_grid(
            [resize(torch.from_numpy(np.asarray(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[i]][:num_ref_concept]],
            nrow=int(num_ref_concept / 1),
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        axs[i + 1, 0].imshow(grid)
        axs[i + 1, 0].set_ylabel(f"\#{topk_ind[i]}")
        axs[i + 1, 0].set_yticks([])
        axs[i + 1, 0].set_xticks([])
    
    plt.tight_layout()
    
    if savename is not None:
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        plt.savefig(savename, dpi=100)

    plt.show()   