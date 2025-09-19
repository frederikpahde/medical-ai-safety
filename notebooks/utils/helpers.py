from sklearn.manifold import TSNE
from umap import UMAP
import h5py
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='umap')

def cosinesim_matrix(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(X) @ torch.nn.functional.normalize(X).t()

def get_2d_data(D, algorithm="umap", metric="euclidean"):
    if algorithm == "umap":
        data_2d = UMAP(metric=metric, min_dist=.1, n_neighbors=15, random_state=42).fit_transform(D)
    elif algorithm == "tsne":
        perp = min(15, D.shape[0]-1)
        data_2d = TSNE(metric='precomputed', init="random", perplexity=perp, random_state=42).fit_transform(D)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return data_2d

def load_all_activations(path_precomputed, classes, layer_name, split):
    vecs = []
    sample_ids = []

    for class_id in classes:
        features_samples = torch.tensor(np.array(
            h5py.File(f"{path_precomputed}/class_{class_id}_all.hdf5")[layer_name]["cavs_max"]
        ))
        metadata = torch.load(f"{path_precomputed}/class_{class_id}_all_meta.pth", weights_only=False)

        if split != "all":
            key_idxs = {"train": "idxs_train", "val": "idxs_val", "test": "idxs_test"}[split]
            idxs_split = metadata[key_idxs]
            features_samples = features_samples[idxs_split]
            samples = np.array(metadata["samples"])[idxs_split]

        vecs.append(features_samples)
        sample_ids.append(samples)

    vecs = torch.cat(vecs, 0).detach().cpu().numpy()
    sample_ids = np.concatenate(sample_ids)

    return vecs, sample_ids