from argparse import ArgumentParser

import h5py
import numpy as np
from datasets import DATASET_CLASSES, load_dataset
from models import get_fn_model_loader, get_canonizer
from utils.dimensionality_reduction import get_2d_data
from utils.plots_2d import get_outlier_label, plot_2d

from utils.helper import load_config, none_or_int, none_or_str
import os
import torch
import tqdm
import matplotlib.pyplot as plt
from crp.image import zimage
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from torchvision.utils import make_grid
import torchvision.transforms as T
from sklearn.neighbors import LocalOutlierFactor
from zennit.composites import EpsilonPlusFlat
from torch.utils.data import DataLoader
from utils.render import vis_opaque_img_border

MAX_NUM_CONCEPTS = 3

def get_parser():
    parser = ArgumentParser(
        description='Plot CRP 2D embedding with outliers.', )

    parser.add_argument('--n', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--class_id', default=1, type=none_or_int)
    parser.add_argument('--new_color_every', default=8, type=int)
    parser.add_argument('--neuron_ids', type=str, 
                        default="480,757,994") 
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--aggr', default="max", type=none_or_str)
    parser.add_argument('--ignore_unused_concepts', default=True, type=bool)
    parser.add_argument('--max_mode', default="relevance", type=str)
    parser.add_argument('--savedir', default="plot_files/crp_outliers", type=str)
    parser.add_argument('--config_file',
                        default="config_files/revealing/isic/local/resnet50d_identity_2.yaml")

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    plot_concept_embedding(config, args.max_mode, args.aggr, args.neuron_ids, args.new_color_every, 
                           args.batch_size, args.split, args.class_id, args.ignore_unused_concepts, args.savedir)


def cosinesim_matrix(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(X) @ torch.nn.functional.normalize(X).t()

def compute_correlation_matrix(X):
    corr = (X - X.mean(0)[None]) / (X.std(0)[None] + 1e-8)
    corr = (corr @ corr.t() / corr.shape[-1]).cpu().abs()
    return corr

def plot_concept_embedding(config, max_mode, aggr, neuron_ids, new_color_every, batch_size, split, 
                           class_id, ignore_unused_concepts, savedir):
    
    model_name = config["model_name"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(DATASET_CLASSES[config["dataset_name"]].classes)
    model = get_fn_model_loader(model_name)(n_class=n_classes,
                                            ckpt_path=config["ckpt_path"]
                                            ).to(device).eval()

    
    dataset = load_dataset(config, normalize_data=True)
    ref_dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": dataset.idxs_train,
        "val": dataset.idxs_val,
        "test": dataset.idxs_test,
        }

    dataset_name = config['dataset_name']
    dataset = dataset if (split is None) or (split=="all") else dataset.get_subset_by_idxs(splits[split])
    ref_dataset = ref_dataset if (split is None) or (split=="all") else ref_dataset.get_subset_by_idxs(splits[split])
    layer_name = config["layer_name"]
    layer_names = [layer_name]

    canonizers = get_canonizer(model_name)
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()
    layer_map = {layer: cc for layer in layer_names}
    print(f"using layer {layer_name}")
    attribution = CondAttribution(model)

    p_artifact = config.get("p_artifact", None)
    artifact_type = config.get("artifact_type", None)
    artifact_extension = f"_{artifact_type}-{p_artifact}" if p_artifact is not None else ""

    path = f"{config['dir_precomputed_data']}/global_relevances_and_activations/{dataset_name}{artifact_extension}/{model_name}"
    
    fv_name = f"{config['dir_precomputed_data']}/crp_files/{dataset_name}_{split}_{model_name}"
    if class_id is not None:
        idxs_class = [i for i in range(len(dataset)) if dataset.get_target(i) == class_id]
        dataset = dataset.get_subset_by_idxs(idxs_class)

    fv = FeatureVisualization(attribution, ref_dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                            path=fv_name, cache=None)
    
    ## Compute / Load Distances
    fname_similarities = f"{config['dir_precomputed_data']}/concept_clustering/{config['dataset_name']}_{split}_{config['model_name']}/similarities/{config['layer_name']}_{class_id}.pth"
    
    if not os.path.isfile(fname_similarities):
        
        dl = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=2)
        
        layer_names = [config["layer_name"]]
        
        with h5py.File(f"{path}/class_{class_id}_{split}.hdf5", 'r') as file:
            layer_data = file[config['layer_name']]
            # data_r = torch.tensor(np.array(layer_data["rel"]))
            data_rmax = torch.tensor(np.array(layer_data["rels_max"]))
            data_rmean = torch.tensor(np.array(layer_data["rels_mean"]))
            data_amax = torch.tensor(np.array(layer_data["cavs_max"]))
            data_amean = torch.tensor(np.array(layer_data["cavs_mean"]))

        os.makedirs(os.path.dirname(fname_similarities), exist_ok=True)

        similarities = {}
        similarities[f"rel_cosine_sim_max"] = cosinesim_matrix(data_rmax.transpose(0,1))
        similarities[f"rel_cosine_sim_avg"] = cosinesim_matrix(data_rmean.transpose(0,1))
        similarities[f"act_cosine_sim_max"] = cosinesim_matrix(data_amax.transpose(0,1))
        similarities[f"act_cosine_sim_avg"] = cosinesim_matrix(data_amean.transpose(0,1))
        similarities["ignore_idxs"] = (data_amax == 0).all(dim=0)

        torch.save(similarities, fname_similarities)

    key_start_dict = {"activation": "act", "relevance": "rel"}
    sim_key = f"{key_start_dict[max_mode]}_cosine_sim"
    sim_key += "_full" if aggr is None else f"_{aggr}"
    print(f"Using similarities with key {sim_key}")
    similarities = torch.load(fname_similarities)
    sim = similarities[sim_key]
    D = (1-sim).clamp(min=0)
    print(f"Computed/Loaded distances, shape: {D.shape}")

    
    for algorithm in ["umap", "tsne"]:
        if ignore_unused_concepts:
            ignore_idxs = similarities["ignore_idxs"]
            D_used = D[~ignore_idxs][:,~ignore_idxs]
            data_2d_used = get_2d_data(D_used, algorithm=algorithm, metric="precomputed")
            data_2d = np.zeros((len(ignore_idxs), 2))
            data_2d[~ignore_idxs] = data_2d_used
        else:
            data_2d = get_2d_data(D, algorithm=algorithm, metric="precomputed")

        if neuron_ids is None:
            # Find outliers in 2d representation
            clf = LocalOutlierFactor(contamination = 0.01, n_neighbors=20)
            _ = clf.fit_predict(data_2d)
            p = torch.tensor(clf.negative_outlier_factor_)
            thresh = -1.0
            top_outlier_idx = p.argsort()
            outlier_concepts = top_outlier_idx[p[top_outlier_idx] < thresh][:MAX_NUM_CONCEPTS].numpy()
        else:
            outlier_concepts = [int(nid) for nid in neuron_ids.split(",")]

        print(f"Potential outliers: {outlier_concepts}")
            
        outlier_labels = [get_outlier_label(x, outlier_concepts, new_color_every) for x in range(0, len(D))]
    
        rf = "vit" not in model_name
        ref_imgs = fv.get_max_reference(outlier_concepts, layer_name, "relevance", (0, 5), composite=composite, rf=rf,
                                    plot_fn=vis_opaque_img_border)


        savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/crp_clustering_{config['dataset_name']}_{config['layer_name']}_{algorithm}_{aggr}_{max_mode}"
        savename += "" if class_id is None else f"_class{class_id}"
        if neuron_ids is not None:
            str_concept_ids = "_".join([str(nid) for nid in outlier_concepts[:3]])
            savename = f"{savename}_{str_concept_ids}"
        os.makedirs(os.path.dirname(savename), exist_ok=True)
        
        if ignore_unused_concepts:
            print(f"Ignoring {ignore_idxs.sum().item()} / {len(ignore_idxs)} unused concepts")
            data_2d = data_2d[~ignore_idxs]
            outlier_labels = np.array(outlier_labels)[~ignore_idxs]
        
        create_plot(data_2d, outlier_labels, outlier_concepts, ref_imgs, algorithm, f"{savename}.pdf")

def show_relmax_refimgs(ref_imgs, axs):
    resize = T.Resize((150, 150))
    for r, (concept, imgs) in enumerate(ref_imgs.items()):
        ax = axs[r]
        grid = make_grid(
        [resize(torch.from_numpy(np.asarray(img)).permute((2, 0, 1))) for img in imgs],
            padding=2)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        ax.imshow(grid)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(f"Concept {concept}")

def create_plot(data, label, concepts, ref_imgs, algorithm, savename):
    nconcepts = len(concepts)
    nrows = 1 + nconcepts + 1
    ncols = 1
    base_size = 1.8
    mul_umap = 5
    gap = .2

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(base_size * mul_umap, (nrows-3) * base_size + base_size * mul_umap), 
                            gridspec_kw={'height_ratios':[mul_umap] + [gap] + [1] * nconcepts})

    ax_umap = axs[0]
    plot_2d(data, label, ax_umap, axis_labels=
            {
                "x": f"{algorithm.upper()} 1",
                "y": f"{algorithm.upper()} 2"
             }
            )
    axs[1].axis("off"); 

    ref_imgs_reduced = {c: ref_imgs[c] for c in concepts}
    axs_relmax = axs[2:2+nconcepts]
    
    show_relmax_refimgs(ref_imgs_reduced, axs_relmax)
    axs_relmax[0].set_title("RelMax")
    
    fig.savefig(savename, bbox_inches="tight")

    ax_umap.axis("off")
    fig.subplots_adjust(hspace=.1)
    fig.savefig(savename[:-4] + "_no_axis" + savename[-4:], bbox_inches="tight")
    
if __name__ == "__main__":
    main()
