import os
import shutil
from argparse import ArgumentParser
import numpy as np
import torch
from PIL import Image
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from tqdm import tqdm
from zennit.composites import EpsilonPlusFlat

from datasets import load_dataset
from models import get_canonizer, get_fn_model_loader
from utils.cav_utils import get_cav_from_model
from utils.helper import load_config

from utils.localization import binarize_heatmaps, get_localizations

MAX_IMGS_SHOW = 6

HARD_CODED_IDS = {
    "ruler": [0,1,2,32,8,33,10],
    "band_aid": [0,2,40,49,50,56],
    "pm": [0,4,17,20,23,25],
    "tube": [1,3,8,18,34,56]
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--split", default="all")
    parser.add_argument("--save_dir", default="plot_files/localization")
    parser.add_argument("--artifact", 
                        # default="ruler")
                        # default="skin_marker")
                        default="band_aid")
                        # default="pm")
                        # default="tube")
    parser.add_argument("--direction_mode", default="svm")
    parser.add_argument("--save_localization", default=True, type=bool)
    parser.add_argument("--save_examples", default=True, type=bool)
    parser.add_argument('--cav_type', type=str, default=None)
    parser.add_argument('--config_file', 
                        # default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")
                        # default="config_files/revealing/isic/local/vgg16_features.22.yaml")
                        default="config_files/revealing/isic/local/vgg16_features.6.yaml")
                        # default="config_files/revealing/chexpert/local/vgg16_binaryTarget-Cardiomegaly_pm_features.22.yaml")
                        # default="config_files/revealing/hyper_kvasir/local/vgg16_features.29.yaml")
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    config = load_config(args.config_file)
    config["direction_mode"] = args.direction_mode
    config["cav_scope"] = None
    if args.artifact == "band_aid":
        config["cav_scope"] = [1] # NV
    elif "attacked" in config["dataset_name"]:
        config["cav_scope"] = config["attacked_classes"]

    localize_artifacts(config,
                       split=args.split,
                       artifact=args.artifact,
                       save_dir=args.save_dir,
                       save_examples=args.save_examples,
                       save_localization=args.save_localization)


def localize_artifacts(config: dict,
                       split: str,
                       artifact: str,
                       save_dir: str,
                       save_examples: bool,
                       save_localization: bool):
    """Spatially localize artifacts in input samples.

    Args:
        config (dict): experiment config
        split (str): data split to use
        artifact (str): artifact
        save_dir (str): save_dir
        save_examples (bool): Store example images
        save_localization (bool): Store localization heatmaps
    """

    dataset_name = config['dataset_name']
    model_name = config['model_name']

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cav_mode = config.get("cav_mode", "cavs_max")
    layer_name = config['layer_name']
    direction_mode = config['direction_mode']
    

    dataset = load_dataset(config, normalize_data=True)

    assert artifact in dataset.sample_ids_by_artifact.keys(), f"Artifact {artifact} unknown."

    artifact_ids = dataset.sample_ids_by_artifact[artifact]

    print(f"Chose {len(artifact_ids)} target samples.")

    model = get_fn_model_loader(model_name=model_name)(n_class=len(dataset.classes),
                                                       ckpt_path=config['ckpt_path'], device=device)
    model = model.to(device)
    model.eval()

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    attribution = CondAttribution(model)

    img_to_plt = lambda x: dataset.reverse_normalization(x.detach().cpu()).permute((1, 2, 0)).int().numpy()
    hm_to_plt = lambda x: x.detach().cpu().numpy()

    ## get CAV
    w = get_cav_from_model(model, dataset, config, artifact, store_cav=True)

    samples = [dataset[i] for i in artifact_ids]
    data_sample = torch.stack([s[0] for s in samples]).to(device)
    batch_size = config["batch_size"]
    num_batches = int(np.ceil(len(data_sample) / batch_size))

    heatmaps = []
    heatmaps_clamped = []
    inp_imgs = []

    for b in tqdm(range(num_batches)):
        data = data_sample[batch_size * b: batch_size * (b + 1)]

        attr, _ = get_localizations(data, w, attribution, composite, config, device)
        inp_imgs.extend([img_to_plt(s.detach().cpu()) for s in data])

        heatmaps.extend([hm_to_plt(h.detach().cpu()) for h in attr.heatmap])
        heatmaps_clamped.extend([hm_to_plt(h.detach().cpu().clamp(min=0)) for h in attr.heatmap])

    heatmaps_binary = binarize_heatmaps(torch.Tensor(heatmaps_clamped), thresholding="otsu").numpy()

    if save_localization:
        savepath = f"data/_localized_artifacts/{config['dataset_name']}/{config['layer_name']}-{direction_mode}/{artifact}"
        save_all_localizations(heatmaps, artifact_ids, savepath)
        savepath = f"data/_localized_artifacts_binary/{config['dataset_name']}/{config['layer_name']}-{direction_mode}/{artifact}"
        save_all_localizations(heatmaps_binary, artifact_ids, savepath)

    if save_examples:
        savepath_hm = f"{save_dir}/cav_heatmaps/{dataset_name}/{model_name}/{artifact}_{layer_name}_{cav_mode}_{direction_mode}"
        savepath_binary = f"{save_dir}/localization_binary/{dataset_name}/{model_name}/{artifact}_{layer_name}_{cav_mode}_{direction_mode}"
        savepath_loc = f"{save_dir}/localization/{dataset_name}/{model_name}/{artifact}_{layer_name}_{cav_mode}_{direction_mode}"
        for ending in ["pdf", "png"]:
            plot_example_figure(inp_imgs, heatmaps_binary, artifact_ids, f"{savepath_binary}.{ending}")
            plot_example_figure(inp_imgs, heatmaps, artifact_ids, f"{savepath_hm}.{ending}")
            plot_example_figure(inp_imgs, heatmaps_clamped, artifact_ids, f"{savepath_loc}.{ending}")

        mname_map = {"resnet50d": "ResNet50",
                     "vgg16": "VGG16"}
        mname = mname_map.get(config['model_name'],config['model_name'])
        savename = f"{save_dir}/examples_all/{config['dataset_name']}/{config['model_name']}/{artifact}_{config['layer_name']}_{split}.pdf"
        
        ids_show = HARD_CODED_IDS.get(artifact, None)
        if ids_show is not None:
            inp_imgs = [inp_imgs[i] for i in ids_show]
            heatmaps_binary = [heatmaps_binary[i] for i in ids_show]
            heatmaps_clamped = [heatmaps_clamped[i] for i in ids_show]
        
        plot_localization(inp_imgs, heatmaps_clamped, heatmaps_binary, 
                            savename, mname, max_imgs=MAX_IMGS_SHOW)   
    
   

def plot_example_figure(inp_imgs, heatmaps, artifact_ids, savepath):
    num_imgs = min(len(inp_imgs), 72) * 2
    grid = int(np.ceil(np.sqrt(num_imgs) / 2) * 2)

    _, axs_ = plt.subplots(grid, grid, dpi=150, figsize=(grid * 1.2, grid * 1.2))

    for j, axs in enumerate(axs_):
        ind = int(j * grid / 2)
        for i, ax in enumerate(axs[::2]):
            if len(inp_imgs) > ind + i:
                ax.imshow(inp_imgs[ind + i])
                ax.set_xlabel(f"sample {int(artifact_ids[ind + i])}", labelpad=1)
            ax.set_xticks([])
            ax.set_yticks([])

        for i, ax in enumerate(axs[1::2]):
            if len(inp_imgs) > ind + i:
                if heatmaps[ind + i].dtype == np.uint8:
                    ax.imshow(heatmaps[ind + i])
                else:
                    max = np.abs(heatmaps[ind + i]).max()
                    ax.imshow(heatmaps[ind + i], cmap="bwr", vmin=-max, vmax=max)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"artifact", labelpad=1)

    plt.tight_layout(h_pad=0.1, w_pad=0.0)
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath)
    plt.show()

def save_all_localizations(heatmaps, artifact_ids, savepath):
    if os.path.isdir(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath, exist_ok=True)
    for i in range(len(heatmaps)):
        sample_id = int(artifact_ids[i])
        heatmap = heatmaps[i]
        heatmap[heatmap < 0] = 0
        heatmap = heatmap / heatmap.max() * 255
        im = Image.fromarray(heatmap).convert("L")
        im.save(f"{savepath}/{sample_id}.png")

def plot_localization(x, hm, loc_pred, savename, model_name, max_imgs=6):
    nrows = 3
    ncols = min(max_imgs, len(x))
    size = 1.7
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*size, nrows*size))
    for i in range(ncols):
        # Input
        ax = axs[0][i]
        ax.imshow(x[i])
        axs[0][0].set_ylabel("Input")

        # HM
        ax = axs[1][i]
        max = np.abs(hm[i]).max()
        ax.imshow(hm[i], cmap="bwr", vmin=-max, vmax=max)
        axs[1][0].set_ylabel("Heatmap")

        # Binary mask
        ax = axs[2][i]
        ax.imshow(loc_pred[i])
        axs[2][0].set_ylabel("Mask (thresh.)")

    for _axs in axs:
        for ax in _axs:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(model_name, fontsize=16,y=0.95)
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    fig.savefig(savename, bbox_inches="tight", dpi=400)

if __name__ == "__main__":
    main()
