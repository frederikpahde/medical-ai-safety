
from argparse import ArgumentParser
import os
import copy
import torch
import numpy as np
from datasets import load_dataset
from models import get_fn_model_loader, get_canonizer
from zennit.composites import EpsilonPlusFlat
import tqdm
from sklearn.metrics import jaccard_score
from crp.attribution import CondAttribution
from utils.cav_utils import get_cav_from_model
from utils.helper import load_config
from utils.localization import binarize_heatmaps, get_localizations
from torch.utils.data import DataLoader
import wandb
from matplotlib import pyplot as plt

MAX_IMGS_SHOW = 5

def get_parser():

    parser = ArgumentParser()
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--artifact", type=str, default="artificial")
    parser.add_argument("--no_wandb", default=False, type=bool)
    parser.add_argument("--config_file", 
                        default="config_files/revealing/isic_attacked_microscope/local/resnet50d_identity_2.yaml")
    parser.add_argument('--savedir', default='plot_files/localization/')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config_file)
    if args.no_wandb:
        config["wandb_api_key"] = None
    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['config_name'], project=config['wandb_project_name'], resume=True)
        print("Initialized WandB")
    run_localization_evaluation(config, args.artifact, args.batch_size, args.savedir)

def run_localization_evaluation(config, artifact, batch_size, savedir):

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.get("device", default_device)

    ## Load Data
    dataset = load_dataset(config, normalize_data=True)
    config_ds_loc = copy.deepcopy(config)
    config_ds_loc['dataset_name'] = f"{config_ds_loc['dataset_name']}_hm"
    config_ds_loc['p_artifact'] = 1
    dataset_loc = load_dataset(config_ds_loc, normalize_data=True)

    ## CAV Scope
    config["cav_scope"] = None
    if artifact == "band_aid":
        config["cav_scope"] = [1] # NV
    elif "attacked" in config["dataset_name"]:
        if "isic" in config["dataset_name"]:
            config["cav_scope"] = [dataset.classes.index(cl) for cl in config["attacked_classes"]]
        else:
            config["cav_scope"] = config["attacked_classes"]

    # Load Model
    model = get_fn_model_loader(config["model_name"])(n_class=len(dataset.classes), ckpt_path=config["ckpt_path"], device=device).to(device)
    model = model.eval()
    attribution = CondAttribution(model)

    canonizers = get_canonizer(config["model_name"])
    if "densenet" in config["model_name"]:
        canonizers[0].apply(model)
        composite = EpsilonPlusFlat()
    else:
        composite = EpsilonPlusFlat(canonizers)

    ## Get CAV 
    cav = get_cav_from_model(model, dataset, config, artifact)

    split_set = {
        "train": dataset_loc.idxs_train,
        "test": dataset_loc.idxs_test
        }
    
    for split in [
        "test",
        # "train",
    ]:
        artifact_ids_split = [i for i in split_set[split] if i in dataset_loc.sample_ids_by_artifact[artifact]]
        dataset_loc_split = dataset_loc.get_subset_by_idxs(artifact_ids_split)
        dl = DataLoader(dataset_loc_split, batch_size=batch_size, shuffle=False)            

        jaccards = []
        artifact_rels = []

        for i, (x, _, loc) in enumerate(tqdm.tqdm(dl)):
            attr, loc_cav = get_localizations(x, cav, attribution, composite, config, device)
            loc_cav_binary = binarize_heatmaps(loc_cav, thresholding="otsu")

            ## IoU (Jaccard) with binarized masks
            jaccards_batch = [jaccard_score(loc[i].reshape(-1).numpy(), loc_cav_binary[i].reshape(-1).numpy()) for i in range(len(loc))]
            jaccards += jaccards_batch

            ## percentage of relevance in artifact region
            inside = (attr.heatmap.detach().cpu().clamp(min=0) * loc).sum((1, 2)) / (
                        attr.heatmap.detach().cpu().clamp(min=0).sum((1, 2)) + 1e-10)
            inside = [val.item() for val in inside]

            artifact_rels += inside

            mname_map = {"resnet50d": "ResNet50",
                         "vgg16": "VGG16"}
            # Create plot for first batch
            if i == 0:
                savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/{artifact}_{config['layer_name']}_{split}.pdf"
                plot_localization(x, loc_cav, loc_cav_binary, loc, dl.dataset, savename, 
                                  mname_map.get(config['model_name'], config['model_name']), max_imgs=MAX_IMGS_SHOW)
                print(f"Saved figure to {savename}")

        metrics_all = {
            f"{split}_{artifact}_perc_artifact_relevance": np.array(artifact_rels).mean(),
            f"{split}_{artifact}_iou": np.array(jaccards).mean(),
        }

        if config.get('wandb_api_key', None):
            wandb.log({**metrics_all, **config})

def plot_localization(x, hm, loc_pred, loc, ds, savename, model_name, max_imgs=6):
    nrows = 4
    ncols = min(max_imgs, len(x))
    size = 1.7
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*size, nrows*size))
    for i in range(ncols):
        # Input
        ax = axs[0][i]
        ax.imshow(ds.reverse_normalization(x[i].detach().cpu()).permute((1, 2, 0)).int().numpy())
        axs[0][0].set_ylabel("Input")

        # Ground truth
        ax = axs[1][i]
        ax.imshow(loc[i].numpy())
        axs[1][0].set_ylabel("Ground Truth")

        # HM
        ax = axs[2][i]
        max = np.abs(hm[i]).max()
        ax.imshow(hm[i], cmap="bwr", vmin=-max, vmax=max)
        # ax.set_title(f"rel: {metric_hm[i]:.2f}")
        axs[2][0].set_ylabel("Heatmap")

        # Binary mask
        ax = axs[3][i]
        ax.imshow(loc_pred[i].numpy())
        # ax.set_title(f"IoU: {metric_loc[i]:.2f}")
        axs[3][0].set_ylabel("Mask (thresh.)")


    for _axs in axs:
        for ax in _axs:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(model_name, fontsize=16,y=0.95)
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    fig.savefig(savename, bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    main()
