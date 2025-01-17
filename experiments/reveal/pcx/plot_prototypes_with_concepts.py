from argparse import ArgumentParser
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

from datasets import load_dataset
from models import get_canonizer, get_fn_model_loader
from utils.helper import get_layer_names_with_identites, load_config
from torchvision.utils import make_grid
from sklearn.mixture import GaussianMixture

from zennit.composites import EpsilonPlusFlat
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.visualization import FeatureVisualization
from crp.image import imgify
import zennit.image as zimage
from utils.render import COLORS, overlay_heatmaps, vis_opaque_img_border
from utils.plots import add_border
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_parser():
    parser = ArgumentParser(
        description='Run CRP preprocessing.', )

    parser.add_argument('--class_id', default=1, type=int)
    parser.add_argument('--n_prototypes', default=4, type=int)
    parser.add_argument('--num_ref_concept', default=4, type=int)
    parser.add_argument('--top_k_sample_prototype', default=4, type=int)
    parser.add_argument('--target_num_concepts', default=6, type=int)
    parser.add_argument('--n_per_prototype', default=2, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--add_border', default=False, type=bool)
    parser.add_argument('--split_ref', default="train", type=str)
    parser.add_argument('--savedir', default="plot_files/pcx", type=str)
    
    parser.add_argument('--config_file',
                        default="config_files/revealing/hyper_kvasir_attacked/local/resnet50d_identity_2.yaml")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    create_prototype_concept_matrix(config, args.class_id, args.n_prototypes, args.num_ref_concept,
                                    args.top_k_sample_prototype, args.n_per_prototype, args.target_num_concepts, 
                                    args.split, args.split_ref, args.add_border, args.savedir)

def create_prototype_concept_matrix(config, 
                                    class_id,
                                    n_prototypes,
                                    num_ref_concept,
                                    top_k_sample_prototype,
                                    n_per_prototype,
                                    target_num_concepts,
                                    split,
                                    split_ref,
                                    do_add_border,
                                    savedir
                                    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = config['dir_precomputed_data']
    dataset_name = config['dataset_name']
    p_artifact = config.get("p_artifact", None)
    artifact_type = config.get("artifact_type", None)
    model_name = config['model_name']
    print(f"Adding border?: {do_add_border}")
    _dataset = load_dataset(config, normalize_data=False)

    splits = {
        "train": _dataset.idxs_train,
        "val": _dataset.idxs_val,
        "test": _dataset.idxs_test,
        }

    dataset = _dataset if (split is None) or (split=="all") else _dataset.get_subset_by_idxs(splits[split])
    dataset_ref = _dataset if (split_ref is None) or (split_ref=="all") else _dataset.get_subset_by_idxs(splits[split_ref])

    artifact_extension = f"_{artifact_type}-{p_artifact}" if p_artifact is not None else ""
    feature_path = f"{results_dir}/global_relevances_and_activations/{dataset_name}{artifact_extension}/{model_name}"
    
    metadata = torch.load(f"{feature_path}/class_{class_id}_{split}_meta.pth")
    sample_ids = np.array(metadata["samples"])
    outputs = metadata["output"]
    classes = np.array([dataset.get_target(i) for i in sample_ids])

    features_samples = torch.tensor(np.array(
        h5py.File(f"{feature_path}/class_{class_id}_{split}.hdf5")[config['layer_name']]["rel"]
    ))

    features_samples = features_samples[outputs.argmax(1) == class_id]
    sample_ids = sample_ids[outputs.argmax(1) == class_id]
    classes = classes[outputs.argmax(1) == class_id]
    indices = sample_ids[classes == class_id]
    features = features_samples[classes == class_id]

    # train gaussian mixture model
    gmm = GaussianMixture(n_components=n_prototypes,
                        random_state=0,
                        covariance_type='full',
                        max_iter=10,
                        verbose=2,
                        reg_covar=1e-6,
                        n_init=1, init_params='kmeans').fit(features)

    distances = np.linalg.norm(features[:, None, :] - gmm.means_, axis=2)
    counts = np.unique(distances.argmin(1), return_counts=True)[1]
    counts_perc = counts / sum(counts) * 100
    prototype_samples = np.argsort(distances, axis=0)[:top_k_sample_prototype]
    prototype_samples = indices[prototype_samples]

    ### PLOTTING CONCEPT MATRIX
    prototypes = torch.from_numpy(gmm.means_)
    top_concepts = torch.topk(prototypes.abs(), n_per_prototype).indices.flatten().unique()

    ## increase number of concepts to consider per prototype to reach total number of target concepts
    if target_num_concepts is not None:
        while len(top_concepts) <= target_num_concepts:
            
            n_per_prototype += 1
            print(f"Increased n_per_prototype to {n_per_prototype}")
            top_concepts = torch.topk(prototypes.abs(), n_per_prototype).indices.flatten().unique()
    
    top_concepts = top_concepts[prototypes[:, top_concepts].abs().amax(0).argsort(descending=True)]
    if target_num_concepts is not None:
        top_concepts = top_concepts[:target_num_concepts]

    concept_matrix = prototypes[:, top_concepts].T

    model = get_fn_model_loader(config["model_name"])(n_class=len(dataset.classes), ckpt_path=config["ckpt_path"]).to(device)
    model.eval()
    canonizers = get_canonizer(config["model_name"])
    composite = EpsilonPlusFlat(canonizers)
    cc = ChannelConcept()

    layer_names = get_layer_names_with_identites(model)
    layer_map = {layer: cc for layer in layer_names}

    print(layer_names)

    attribution = CondAttribution(model)

    fv_name = f"{config['dir_precomputed_data']}/crp_files/{config['dataset_name']}_{split_ref}_{config['model_name']}"
    fv_ref = FeatureVisualization(attribution, dataset_ref, layer_map, preprocess_fn=dataset_ref.normalize_fn,
                              path=fv_name, cache=None)
    
    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=dataset.normalize_fn,
                              path=fv_name, cache=None)


    topk_ind = [int(x) for x in top_concepts]

    rf = "vit" not in config["model_name"]
    ref_imgs = fv_ref.get_max_reference(topk_ind, config['layer_name'], "relevance", (0, num_ref_concept), composite=composite, rf=rf,
                                    plot_fn=vis_opaque_img_border)

    layer_name = config["layer_name"]
    savename = f"{savedir}/{config['dataset_name']}/{config['model_name']}/prototype_data/pcx_{config['dataset_name']}_{class_id}_prototypes_with_concepts_{layer_name}"
    
    num_hms = 2
    top_concepts_by_prototype = top_concepts[torch.topk(concept_matrix.T, num_hms).indices]
    cond_heatmaps_prototypes = torch.stack([get_cond_heatmaps_prototype(fv, model, prototype_samples[:, pid], top_concepts_by_prototype[pid],
                                                                        attribution, layer_name, composite) 
                                            for pid in range(n_prototypes)])

    # plot_matrix(concept_matrix, dataset, prototype_samples, ref_imgs, cond_heatmaps_prototypes, top_concepts_by_prototype,
    #             counts, counts_perc, topk_ind, num_hms, f"{savename}_overlay.png")
    
    plot_matrix(concept_matrix, dataset, prototype_samples, ref_imgs, cond_heatmaps_prototypes, top_concepts_by_prototype,
                counts, counts_perc, topk_ind, num_ref_concept, top_k_sample_prototype, 0, do_add_border, f"{savename}.pdf")

def get_cond_heatmaps_given_sample_id(fv, model, sid, concept_ids, attribution, layer_name, composite):
    # First forward pass to get prediction
    data_p, y = fv.get_data_sample(sid, preprocessing=True)
    pred = model(data_p).detach().argmax().cpu()
    target = pred
    
    # Define conditions
    conditions = [{"y": target,
                   layer_name: c_id
                   } for c_id in concept_ids]
    data_p, _ = fv.get_data_sample(sid, preprocessing=True)
    
    # Second forward pass with conditions
    attr_cond = attribution(data_p.clone().requires_grad_(), conditions, composite)
    attribution.model.zero_grad()
    cond_heatmap, _, _, _ = attr_cond
    torch.cuda.empty_cache()
    
    return cond_heatmap.detach().cpu()

def get_cond_heatmaps_prototype(fv, model, sample_ids, top_concepts, attribution, layer_name, composite):
    return torch.stack([get_cond_heatmaps_given_sample_id(fv, model, sid, top_concepts, attribution, layer_name, composite) for sid in sample_ids])

def plot_matrix(concept_matrix, dataset, prototype_samples, ref_imgs, cond_heatmaps_prototypes, top_concepts_by_prototype,
                counts, counts_perc, topk_ind, num_ref_concept, top_k_sample_prototype, num_hms, do_add_border, savename):
    n_concepts, n_prototypes = concept_matrix.shape

    fig, axs = plt.subplots(nrows=n_concepts + 1, ncols=n_prototypes + 1, figsize=(n_prototypes + num_ref_concept, 
                                                                                   n_concepts + top_k_sample_prototype), dpi=150,
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

            # Highlight top concepts
            if topk_ind[i] in top_concepts_by_prototype[j]:
                cid = np.where(top_concepts_by_prototype[j] == topk_ind[i])[0][0]
                if cid < num_hms:
                    ax = axs[i+1][j+1]
                    ax.axis("on")
                    ax.set_yticks([]), ax.set_xticks([])
                    for spine in ax.spines.values():
                        spine.set_edgecolor(COLORS[cid])  # Change border color
                        spine.set_linewidth(5) 

    resize = T.Resize((120, 120))
    for i in range(n_prototypes):
        grid = make_grid(
            [resize(
                _add_border(overlay_heatmaps(dataset[prototype_samples[j][i]][0], 
                                             [cond_heatmaps_prototypes[i][j][cid] for cid in range(num_hms)], 
                                             COLORS[:num_hms]),
                                             prototype_samples[j][i], dataset, do_add_border)
                # dataset[prototype_samples[j][i]][0]
                )
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
            [resize(torch.from_numpy(np.asarray(i)).permute((2, 0, 1))) for i in ref_imgs[topk_ind[i]]],
            nrow=int(num_ref_concept / 1),
            padding=0)
        grid = np.array(zimage.imgify(grid.detach().cpu()))
        axs[i + 1, 0].imshow(grid)
        axs[i + 1, 0].set_ylabel(f"concept {topk_ind[i]}")
        axs[i + 1, 0].set_yticks([])
        axs[i + 1, 0].set_xticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(savename), exist_ok=True)
    plt.savefig(savename, dpi=300)

    plt.show()    

def _add_border(img, sample_id, dataset, do_add_border):
    img = (img * 255).type(torch.uint8)
    col_art = (torch.Tensor(sns.color_palette()[1]) * 255).type(torch.uint8)
    w = 10
    if do_add_border:
        artifact_ids = dataset.sample_ids_by_artifact[list(dataset.sample_ids_by_artifact.keys())[0]]
        if sample_id in artifact_ids:
            img = add_border(img, col_art, w)
    return img

if __name__ == "__main__":
    main()
