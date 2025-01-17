import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from zennit import image as zimage
from datasets import load_dataset
from experiments.evaluation.prepare_for_evaluation import prepare_model_for_evaluation
from models import get_fn_model_loader
from utils.helper import load_config
from captum.attr import ShapleyValueSampling
from skimage.segmentation import mark_boundaries 
from fast_slic import Slic

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
                        default="config_files/bias_mitigation_controlled/hyper_kvasir_attacked/local/vit_b_16_torchvision_RRClarc_lamb100_sgd_lr0.001_inspection_layer.yaml")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    config = load_config(args.config_file)

    if config.get('wandb_api_key', None):
        os.environ["WANDB_API_KEY"] = config['wandb_api_key']
        wandb.init(id=config['wandb_id'], project=config['wandb_project_name'], resume=True)

    sample_ids = [int(i) for i in args.sample_ids.split(",")] if args.sample_ids else None

    plot_corrected_model(config, sample_ids, args.normalized, args.plot_to_wandb, args.results_dir)


def plot_corrected_model(config, sample_ids, normalized, plot_to_wandb, path):
    dataset_name = config['dataset_name']
    config_name = config['config_name']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(config, normalize_data=True)
    
    if sample_ids is None:
        # only "corrected" artifact
        sample_ids = dataset.sample_ids_by_artifact[config['artifact']][:8]

    data = torch.stack([dataset[j][0] for j in sample_ids], dim=0)
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

    slic = Slic(num_components=100, compactness=10)
    svs_original = ShapleyValueSampling(model_original)
    svs_corrected = ShapleyValueSampling(model_corrected)

    assignments_batch = None
    for j in range(len(data)):
        sample_np = dataset.reverse_normalization(data[j]).numpy().transpose(1,2,0).astype("uint8")
        assignment = torch.tensor(slic.iterate(sample_np.copy(order='C'))).to(device).unsqueeze(0)
        assignments_batch = assignment if assignments_batch is None else torch.cat([assignments_batch, assignment])

    attr_original = svs_original.attribute(data.to(device), target=target.to(device), 
                                n_samples=150, feature_mask=assignments_batch.unsqueeze(1), 
                                show_progress=True, baselines=0)
    
    hm_original = attr_original.sum(1).detach()
    
    attr_corrected = svs_corrected.attribute(data.to(device), target=target.to(device), 
                                n_samples=150, feature_mask=assignments_batch.unsqueeze(1), 
                                show_progress=True, baselines=0)
    
    hm_corrected = attr_corrected.sum(1).detach()

    max_corrected = get_normalization_constant(hm_corrected, normalized)
    max_original = get_normalization_constant(hm_original, normalized)

    joint_max = get_joined_max(max_corrected, max_original)

    hm_corrected = hm_corrected / joint_max
    hm_corrected = hm_corrected.cpu().numpy()

    hm_original = hm_original / joint_max
    hm_original = hm_original.cpu().numpy()

    heatmaps_diff = hm_corrected - hm_original
    if normalized == "max":
        heatmaps_diff /= heatmaps_diff.reshape(heatmaps_diff.shape[0], -1).max(1)[:, None, None]
    elif normalized == "abs_max":
        heatmaps_diff /= np.abs(heatmaps_diff).reshape(heatmaps_diff.shape[0], -1).max(1)[:, None, None]
    # plot input images and heatmaps in grid
    size = 2
    fig, axs = plt.subplots(5, len(sample_ids), figsize=(len(sample_ids) * size, 3 * size), dpi=300)

    for i, sample_id in enumerate(sample_ids):
        img = dataset.reverse_normalization(dataset[sample_id][0]).permute(1, 2, 0).int().numpy()
        axs[0, i].imshow(img)

        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[0, i].set_title(f"Sample {sample_id}")

        assignment_np = assignments_batch[i].detach().cpu().numpy()
        img_with_assignment = mark_boundaries(img / 255, assignment_np)
        axs[1, i].imshow(img_with_assignment)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])

        axs[2, i].imshow(hm_original[i], vmin=-1, vmax=1, cmap="bwr")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])

        axs[3, i].imshow(hm_corrected[i], vmin=-1, vmax=1, cmap="bwr")
        axs[3, i].set_xticks([])
        axs[3, i].set_yticks([])

        axs[4, i].imshow(zimage.imgify(heatmaps_diff[i], vmin=-1., vmax=1., level=1.0, cmap='coldnhot'))
        axs[4, i].set_xticks([])
        axs[4, i].set_yticks([])

        # make border thicker
        for ax in axs[:, i]:
            for spine in ax.spines.values():
                spine.set_linewidth(2)

        # set label for the first column
        if i == 0:
            axs[0, i].set_ylabel("Input")
            axs[1, i].set_ylabel("Features")
            axs[2, i].set_ylabel("Vanilla")
            axs[3, i].set_ylabel(str(config['method']))
            axs[4, i].set_ylabel("Difference")

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
    plt.savefig(f"{path}/{config['wandb_id']}_no_labels.png", bbox_inches="tight", dpi=400)
    plt.show()

    # log png to wandb
    if plot_to_wandb:
        wandb.log({"corrected_model": wandb.Image(f"{path}/{config['wandb_id']}.jpeg")})

    print("Done.")


def get_normalization_constant(hm, normalization_mode):
    if normalization_mode == 'max':
        return hm.flatten(start_dim=1).max(1, keepdim=True).values[:, None]
    elif normalization_mode == 'abs_max':
        return hm.flatten(start_dim=1).abs().max(1, keepdim=True).values[:, None]
    else:
        raise ValueError("Unknown normalization")

def get_joined_max(max1, max2):
    return torch.max(max1,max2)

if __name__ == "__main__":
    main()
