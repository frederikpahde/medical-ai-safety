from sklearn.manifold import TSNE
from umap import UMAP
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
