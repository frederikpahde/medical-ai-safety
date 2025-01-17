import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from crp.attribution import CondAttribution
from matplotlib import pyplot as plt
from zennit.composites import EpsilonPlusFlat
from zennit import image as zimage
from datasets import load_dataset
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader, get_canonizer
from utils.helper import load_config

torch.random.manual_seed(0)
np.random.seed(0)
random.seed(0)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--sample_ids", default=None, type=str)
    parser.add_argument("--normalized", default="max", type=str)
    parser.add_argument("--results_dir", default="results/plot_corrected_model", type=str)
    parser.add_argument('--plot_to_wandb', default=True, type=bool)
    parser.add_argument('--config_file',
                        default="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/local/resnet50d_RRClarc_lamb1000000_adam_lr0.001_identity_2.yaml")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=f"{config['wandb_project_name']}", resume=True)

    sample_ids = [int(i) for i in args.sample_ids.split(",")] if args.sample_ids else None

    plot_corrected_model(config, sample_ids, args.normalized, args.plot_to_wandb, args.results_dir)


def plot_corrected_model(config, sample_ids, normalized, plot_to_wandb, path):
    dataset_name = config['dataset_name']
    config_name = config['config_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(config, normalize_data=True)

    if sample_ids is None:
        # only "corrected" artifact
        sample_ids = dataset.sample_ids_by_artifact[config['artifact']][:5]

    data = torch.stack([dataset[j][0] for j in sample_ids], dim=0).to(device)

    target = torch.stack([dataset[j][1] for j in sample_ids], dim=0)
    ckpt_path_corrected = f"{config['checkpoint_dir_corrected']}/{config_name}/last.ckpt"
    if config["num_epochs"] == 0 and dataset_name == "imagenet":
        ckpt_path_corrected = None
    ckpt_path_original = config['ckpt_path']
    model_corrected = get_fn_model_loader(model_name=config['model_name'])(n_class=len(dataset.classes),
                                                                           ckpt_path=ckpt_path_corrected, device=device)
    model_corrected = prepare_model_for_evaluation(model_corrected, dataset, device, config)

    model_original = get_fn_model_loader(model_name=config['model_name'])(n_class=len(dataset.classes),
                                                                           ckpt_path=ckpt_path_original, device=device)

    model_original.eval()
    model_original = model_original.to(device)

    attribution_corrected = CondAttribution(model_corrected)
    attribution_original = CondAttribution(model_original)
    canonizers = get_canonizer(config['model_name'])
    composite = EpsilonPlusFlat(canonizers)

    condition = [{"y": c_id.item()} for c_id in target]
    attr_corrected = attribution_corrected(data.requires_grad_(), condition, composite)

    # max = get_normalization_constant(attr_corrected, config['normalized'])

    # computed corrupted heatmaps
    condition_original = [{"y": c_id.item()} for c_id in target]
    attr_original = attribution_original(data.requires_grad_(), condition_original, composite)

    max_corrected = get_normalization_constant(attr_corrected, normalized)
    max_original = get_normalization_constant(attr_original, normalized)

    joint_max = get_joined_max(max_corrected, max_original)

    heatmaps_corrected = attr_corrected.heatmap / joint_max
    heatmaps_corrected = heatmaps_corrected.detach().cpu().numpy()

    heatmaps_original = attr_original.heatmap / joint_max
    heatmaps_original = heatmaps_original.detach().cpu().numpy()

    heatmaps_diff = heatmaps_corrected - heatmaps_original
    if normalized == "max":
        heatmaps_diff /= heatmaps_diff.reshape(heatmaps_diff.shape[0], -1).max(1)[:, None, None]
    elif normalized == "abs_max":
        heatmaps_diff /= np.abs(heatmaps_diff).reshape(heatmaps_diff.shape[0], -1).max(1)[:, None, None]
    # plot input images and heatmaps in grid
    size = 1.5
    fig, axs = plt.subplots(4, len(sample_ids), figsize=(len(sample_ids) * size, 4 * size), dpi=300)

    for i, sample_id in enumerate(sample_ids):
        axs[0, i].imshow(dataset.reverse_normalization(dataset[sample_id][0]).permute(1, 2, 0) / 255)

        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        # axs[0, i].set_title(f"Sample {sample_id}")
        # axs[0, i].axis("off")

        axs[1, i].imshow(heatmaps_original[i], vmin=-1, vmax=1, cmap="bwr")
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        # axs[1, i].axis("off")

        axs[2, i].imshow(heatmaps_corrected[i], vmin=-1, vmax=1, cmap="bwr")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        # axs[2, i].axis("off")

        axs[3, i].imshow(zimage.imgify(heatmaps_diff[i], vmin=-1., vmax=1., level=1.0, cmap='coldnhot'))
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])
        # axs[3, i].axis("off")

        # make border thicker
        for ax in axs[:, i]:
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # set label for the first column
        if i == 0:
            axs[0, i].set_ylabel("Input")
            axs[1, i].set_ylabel("Vanilla")
            axs[2, i].set_ylabel(stringify_method(str(config['method'])))
            axs[3, i].set_ylabel("Difference")

    plt.tight_layout()

    # save figure with and without labels as pdf
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f"{path}/{config['wandb_id']}.png", bbox_inches="tight", dpi=400)
    plt.savefig(f"{path}/{config['wandb_id']}.jpeg", bbox_inches="tight", dpi=200)

    # disable labels
    for ax in axs.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # ax.axis("off")
    plt.savefig(f"{path}/{config['wandb_id']}_no_labels.png", bbox_inches="tight", dpi=400)
    plt.show()

    # log png to wandb
    if plot_to_wandb:
        wandb.log({"corrected_model": wandb.Image(f"{path}/{config['wandb_id']}.png")})

    print("Done.")

def stringify_method(method):
    if "rrr" in method.lower():
        return "RRR"
    elif "rrclarc" in method.lower():
        return "RR-ClArC"
    elif "pclarc" in method.lower():
        return "P-ClArC"
    else:
        return method

def get_normalization_constant(attr, normalization_mode):
    if normalization_mode == 'max':
        return attr.heatmap.flatten(start_dim=1).max(1, keepdim=True).values[:, None]
    elif normalization_mode == 'abs_max':
        return attr.heatmap.flatten(start_dim=1).abs().max(1, keepdim=True).values[:, None]
    else:
        raise ValueError("Unknown normalization")

def get_joined_max(max1, max2):
    return torch.max(max1,max2)

if __name__ == "__main__":
    main()
