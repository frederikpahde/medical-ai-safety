
from argparse import ArgumentParser
import os
import copy
import torch
import numpy as np
from datasets import load_dataset
from models import get_fn_model_loader
import tqdm
from sklearn.metrics import jaccard_score
from utils.cav_utils import get_cav_from_model
from utils.helper import load_config
from utils.localization import binarize_heatmaps
from torch.utils.data import DataLoader
import wandb
from matplotlib import pyplot as plt
from captum.attr import ShapleyValueSampling
from skimage.segmentation import mark_boundaries 
from fast_slic import Slic

MAX_IMGS_SHOW = 5

def get_parser():

    parser = ArgumentParser()
    
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--artifact", type=str, default="artificial")
    parser.add_argument("--no_wandb", default=True, type=bool)
    parser.add_argument("--config_file", 
                        default="config_files/revealing/hyper_kvasir_attacked/local/vit_b_16_torchvision_inspection_layer.yaml")
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

    ## Get CAV 
    cav = get_cav_from_model(model, dataset, config, artifact)

    ## Overwrite model
    in_features = model.heads.head.in_features
    new_head = torch.nn.Linear(in_features=in_features, out_features=1, bias=False)
    new_head.weight.data = cav[None, :].to(device)
    model.heads.head = new_head

    svs = ShapleyValueSampling(model)


    split_set = {
        "train": dataset_loc.idxs_train,
        "test": dataset_loc.idxs_test
        }
    slic = Slic(num_components=100, compactness=10)

    for split in [
        "test",
        # "train",
        
    ]:
        idxs_split = split_set[split]
        artifact_ids_split = [i for i in idxs_split if i in dataset_loc.sample_ids_by_artifact[artifact]]
        dataset_loc_split = dataset_loc.get_subset_by_idxs(artifact_ids_split)
        dl = DataLoader(dataset_loc_split, batch_size=batch_size, shuffle=False)            

        jaccards = []
        artifact_rels = []
        all_scores = []

        for i, (x, _, loc) in enumerate(tqdm.tqdm(dl)):

            assignments_batch = None
            for j in range(len(x)):
                sample_np = dataset_loc_split.reverse_normalization(x[j]).numpy().transpose(1,2,0).astype("uint8")
                assignment = torch.tensor(slic.iterate(sample_np.copy(order='C'))).to(device).unsqueeze(0)
                assignments_batch = assignment if assignments_batch is None else torch.cat([assignments_batch, assignment])

            scores = model(x.to(device)).detach().cpu().numpy().reshape(-1)
            all_scores.append(scores)
            attr_batch = svs.attribute(x.to(device), target=None, 
                                n_samples=150, feature_mask=assignments_batch.unsqueeze(1), 
                                show_progress=True, baselines=0)

            loc_cav = attr_batch.squeeze(1).sum(1).detach().cpu().clamp(min=0)
            loc_cav_binary = binarize_heatmaps(loc_cav, thresholding="otsu")

            ## IoU (Jaccard) with binarized masks
            jaccards_batch = [jaccard_score(loc[i].reshape(-1).numpy(), loc_cav_binary[i].reshape(-1).numpy()) for i in range(len(loc))]
            jaccards += jaccards_batch

            ## percentage of relevance in artifact region
            inside = (loc_cav * loc).sum((1, 2)) / (
                    loc_cav.sum((1, 2)) + 1e-10)
            inside = [val.item() for val in inside]

            artifact_rels += inside

            mname_map = {"resnet50d": "ResNet50",
                         "vgg16": "VGG16",
                         "vit_b_16_torchvision": "ViT"}
            # Create plot for first batch
            if i == 0:
                savename = f"{savedir}/quantitative/{config['dataset_name']}/{config['model_name']}/{artifact}_{config['layer_name']}_{split}.pdf"
                plot_localization(x, assignments_batch, loc_cav, loc_cav_binary, loc, 
                                  dl.dataset, savename, mname_map.get(config['model_name'],config['model_name']), max_imgs=MAX_IMGS_SHOW)
                print(f"Saved figure to {savename}")

        all_scores = np.concatenate(all_scores)
        print(f"Scores: {all_scores.mean()} (mean), {(all_scores > 0).sum()} (>0)")
        metrics_all = {
            f"{split}_{artifact}_perc_artifact_relevance": np.array(artifact_rels).mean(),
            f"{split}_{artifact}_iou": np.array(jaccards).mean(),
        }

        if config.get('wandb_api_key', None):
            print(f"Logging metrics: {metrics_all}")
            wandb.log({**metrics_all, **config})

def plot_localization(x, assignments_batch, hm, loc_pred, loc, ds, savename, model_name, max_imgs=6):
    nrows = 5
    ncols = min(max_imgs, len(x))
    size = 1.7
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*size, nrows*size))
    for i in range(ncols):
        # Input
        ax = axs[0][i]
        img = ds.reverse_normalization(x[i].detach().cpu()).permute((1, 2, 0)).int().numpy()
        ax.imshow(img)
        # ax.set_title(f"{scores[i]:.2f}")
        axs[0][0].set_ylabel("Input")

        ax = axs[1][i]
        assignment_np = assignments_batch[i].detach().cpu().numpy()
        img_with_assignment = mark_boundaries(img / 255, assignment_np)
        ax.imshow(img_with_assignment)
        axs[1][0].set_ylabel("Features")

        # Ground truth
        ax = axs[2][i]
        ax.imshow(loc[i].numpy())
        axs[2][0].set_ylabel("Ground Truth")
        

        # HM
        ax = axs[3][i]
        max = np.abs(hm[i]).max()
        ax.imshow(hm[i], cmap="bwr", vmin=-max, vmax=max)
        # ax.set_title(f"rel: {metric_hm[i]:.2f}")
        axs[3][0].set_ylabel("Heatmap")

        # Binary mask
        ax = axs[4][i]
        ax.imshow(loc_pred[i].numpy())
        # ax.set_title(f"IoU: {metric_loc[i]:.2f}")
        axs[4][0].set_ylabel("Mask (thresh.)")


    for _axs in axs:
        for ax in _axs:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(model_name, fontsize=16,y=0.95)
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    fig.savefig(savename, bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    main()
