import torch
import numpy as np
from sklearn.mixture import GaussianMixture


def run_pcx(features_samples,
            outputs, 
            classes,
            sample_ids,
            class_id=0,
            n_prototypes=4,
            top_k_sample_prototype=4,
            n_per_prototype=2):
    
    features_samples = features_samples[outputs.argmax(1) == class_id]
    sample_ids = sample_ids[outputs.argmax(1) == class_id]
    classes = classes[outputs.argmax(1) == class_id]
    indices = sample_ids[classes == class_id]
    features = features_samples[classes == class_id]

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

    ### PREPARE CONCEPT MATRIX
    prototypes = torch.from_numpy(gmm.means_)
    top_concepts = torch.topk(prototypes.abs(), n_per_prototype).indices.flatten().unique()
    top_concepts = top_concepts[prototypes[:, top_concepts].abs().amax(0).argsort(descending=True)]
    concept_matrix = prototypes[:, top_concepts].T
    topk_ind = [int(x) for x in top_concepts]

    return concept_matrix, prototype_samples, counts, counts_perc, topk_ind