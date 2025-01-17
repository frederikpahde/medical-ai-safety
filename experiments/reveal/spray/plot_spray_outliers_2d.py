from argparse import ArgumentParser
import numpy as np
from datasets import load_dataset
import h5py
from utils.helper import load_config, none_or_int
from utils.plots_2d import get_outlier_label, plot_2d
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.utils import make_grid
from sklearn.neighbors import LocalOutlierFactor

MAX_NUM_SAMPLES = 24

SAMPLE_IDS = [
    1990, 2015, 2047, 2120, 2122, 2128, 2149, 2150, 2189, 2220, 2230, 2234, 2252, 2313, 2336, 2391
    ]
SAMPLE_IDS_STR = ",".join([str(sid) for sid in SAMPLE_IDS])
print(f"plotting {len(SAMPLE_IDS)} sample_ids {SAMPLE_IDS_STR}")

def get_parser():
    parser = ArgumentParser(
        description='Plot SpRAy 2D embedding with outliers.', )

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--class_id', default=1, type=none_or_int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--outlier_sample_ids', default=SAMPLE_IDS_STR, type=str)
    parser.add_argument('--new_color_every', default=18, type=int)
    parser.add_argument('--savedir', default="plot_files/spray_outliers", type=str)
    parser.add_argument('--config_file',
                        default="config_files/revealing/hyper_kvasir_attacked/local/resnet50d_identity_2.yaml"
                        )
   

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    plot_spray_embedding(config, args.split, args.class_id, args.outlier_sample_ids, args.new_color_every, args.savedir)


def plot_spray_embedding(config, split, class_id, outlier_sample_ids, new_color_every, savedir):
    
    
    dataset = load_dataset(config, normalize_data=True)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])
    
    results_dir = config['dir_precomputed_data']
    analysis_file = f"{results_dir}/spray/{config['dataset_name']}/{config['config_name']}_{split}.hdf5"
    
    for emb_type in ["umap", "tsne"]:
        with h5py.File(analysis_file, 'r') as fp:
            sample_ids = np.array(fp[str(class_id)]['index'])
            data_2d = fp[str(class_id)]['embedding'][emb_type][::1]
        
        if outlier_sample_ids is None:
            # Find outliers in 2d representation
            clf = LocalOutlierFactor(contamination = 0.01, n_neighbors=20)
            _ = clf.fit_predict(data_2d)
            p = torch.tensor(clf.negative_outlier_factor_)
            thresh = -1.0
            top_outlier_idx = p.argsort()
            outlier_samples = top_outlier_idx[p[top_outlier_idx] < thresh][:MAX_NUM_SAMPLES].numpy()
        else:
            outlier_samples_full = [int(sid) for sid in outlier_sample_ids.split(",")]
            sample_id_map = {id_orig: i for i, id_orig in enumerate(sample_ids)}
            outlier_samples = [sample_id_map[outlier_id] for outlier_id in outlier_samples_full if outlier_id in sample_id_map]

        outlier_samples = outlier_samples[:MAX_NUM_SAMPLES]

        print(f"Potential outliers: {outlier_samples}")    
                
        outlier_labels = [get_outlier_label(x, outlier_samples, new_color_every) for x in range(0, len(data_2d))]

        savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/spray_{config['dataset_name']}_{config['layer_name']}_{emb_type}"
        savename += "" if class_id is None else f"_class{class_id}"
        if outlier_sample_ids is not None:
            str_concept_ids = "_".join([str(nid) for nid in outlier_samples_full[:3]])
            savename = f"{savename}_{str_concept_ids}"
        os.makedirs(os.path.dirname(savename), exist_ok=True)

        outlier_imgs = [dataset.reverse_normalization(dataset[i][0]) for i in outlier_samples_full]
        print(f"Store as {savename}")
        create_plot(data_2d, outlier_labels, outlier_imgs[:MAX_NUM_SAMPLES], emb_type, f"{savename}.pdf")

def show_outlier_imgs(imgs, ax):
    grid = make_grid([img for img in imgs], padding=2, nrow=6)
    ax.imshow(grid.numpy().transpose(1,2,0))
    ax.set_yticks([]); ax.set_xticks([])

def create_plot(data, label, outlier_imgs, algorithm, savename):

    nrows = 3 
    ncols = 1
    base_size = 1.8
    mul_umap = 5
    gap = .1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(base_size * mul_umap, base_size * mul_umap * 1.5), 
                            gridspec_kw={'height_ratios':[mul_umap] + [gap] + [mul_umap / 2]})

    ax_umap = axs[0]
    plot_2d(data, label, ax_umap, axis_labels=
            {
                "x": f"{algorithm.upper()} 1",
                "y": f"{algorithm.upper()} 2"
             }
            )
    axs[1].axis("off"); 

    show_outlier_imgs(outlier_imgs, axs[2])
    fig.savefig(savename, bbox_inches="tight", dpi=300)
    ax_umap.axis("off")
    fig.savefig(savename[:-4] + f"_no_axis.{savename[-3:]}", bbox_inches="tight", dpi=300)
    
if __name__ == "__main__":
    main()
