import os
from argparse import ArgumentParser
import h5py
import numpy as np
import torch
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.affinity import SparseKNN
from corelay.processor.clustering import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from corelay.processor.embedding import TSNEEmbedding, UMAPEmbedding, EigenDecomposition
from corelay.processor.flow import Parallel

from experiments.reveal.spray.spray_utils import VARIANTS, csints
from utils.helper import load_config

"""
This python script is based on the examples from https://github.com/virelay/corelay.
"""

def get_parser():
    parser = ArgumentParser(
        description='Run CRP preprocessing.', )

    parser.add_argument('--variant', default="spectral")
    parser.add_argument('--split', default="train")
    parser.add_argument('--class_indices', type=csints, default="1")
    parser.add_argument('--n_eigval', type=int, default=32)
    parser.add_argument('--n_clusters', type=csints, default=','.join(str(elem) for elem in range(1, 2)))
    parser.add_argument('--n_neighbors', default=32)
    parser.add_argument('--corrected_model', type=bool, default=False)
    parser.add_argument('--config_file',
                        default="notebooks/r2r_config_vgg16.json"
                        )
   

    return parser

def main():
    args = get_parser().parse_args()
    config = load_config(args.config_file)
    run_spray(config, args.variant, args.split, args.class_indices,
              args.n_eigval, args.n_clusters, args.n_neighbors, args.corrected_model)

def run_spray(config,
              variant, 
              split,
              class_indices, 
              n_eigval, 
              n_clusters, 
              n_neighbors,
              corrected_model=False):
    
    preprocessing = VARIANTS[variant]['preprocessing']
    distance = VARIANTS[variant]['distance']

    pipeline = SpectralClustering(
        preprocessing=preprocessing,
        pairwise_distance=distance,
        affinity=SparseKNN(n_neighbors=n_neighbors, symmetric=True),
        embedding=EigenDecomposition(n_eigval=n_eigval, is_output=True),
        clustering=Parallel([
            Parallel([
                KMeans(n_clusters=k) for k in n_clusters
            ], broadcast=True),
            Parallel([
                DBSCAN(eps=k / 10.) for k in n_clusters
            ], broadcast=True),
            HDBSCAN(),
            Parallel([
                AgglomerativeClustering(n_clusters=k) for k in n_clusters
            ], broadcast=True),
            Parallel([
                UMAPEmbedding(),
                TSNEEmbedding(),
            ], broadcast=True)
        ], broadcast=True, is_output=True)
    )

    results_dir = config['dir_precomputed_data']
    model_name = config["model_name"]
    dataset_name = config["dataset_name"]
    p_artifact = config.get("p_artifact", None)
    artifact_type = config.get("artifact_type", None)
    artifact_extension = f"_{artifact_type}-{p_artifact}" if p_artifact is not None else ""

    if corrected_model:
        # path = f"{results_dir}/global_relevances_and_activations/{config['config_name']}"
        path = f"{results_dir}/global_relevances_and_activations/{dataset_name}/{model_name}"
        extra_part = config['method']
        if "clarc" in config['method'].lower():
            extra_part += f"_{config['direction_mode']}"
        if "lamb" in config:
            extra_part += f"_{config['lamb']}"
        path += f"_{extra_part}"
    else:
        path = f"{results_dir}/global_relevances_and_activations/{dataset_name}{artifact_extension}/{model_name}"
        
    layer = config["layer_name"]
    mode = "hm" if layer == "input_identity" else "rel"

    for i, class_index in enumerate(class_indices):
        print(f"Loading class {class_index}")
        data = torch.tensor(np.array(
            h5py.File(f"{path}/class_{class_index}_all.hdf5")[config['layer_name']][mode]
        ))
        metadata = torch.load(f"{path}/class_{class_index}_all_meta.pth", weights_only=False)
        sample_ids_by_split = {
            "all": np.array(metadata["samples"]),
            "train": np.array(metadata["samples_train"]),
            "val": np.array(metadata["samples_val"]),
            "test": np.array(metadata["samples_test"])
            }
        sample_ids = sample_ids_by_split[split]
        if split != "all":
            idxs = {
                "train": metadata['idxs_train'],
                "val": metadata['idxs_val'],
                "test": metadata['idxs_test']
                }[split]
            data = data[idxs]
        train_flag = None
        print(f"Shape of Data: {data.shape}")
        print(f"Computing class {class_index}")
        (eigenvalues, embedding), (kmeans, dbscan, hdbscan, agglo, (umap, tsne)) = pipeline(data)

        print(f"Saving class {class_index}")
        analysis_file = f"{results_dir}/spray/{config['dataset_name']}/{config['config_name']}_{split}.hdf5"
        os.makedirs(os.path.dirname(analysis_file), exist_ok=True)

        m = "w" if i == 0 else "a"
        with h5py.File(analysis_file, m) as fp:
            analysis_name = f"{class_index}"
            g_analysis = fp.require_group(analysis_name)
            g_analysis['index'] = sample_ids

            g_embedding = g_analysis.require_group('embedding')
            g_embedding['spectral'] = embedding.astype('float32')
            g_embedding['spectral'].attrs['eigenvalue'] = eigenvalues.astype('float32')

            g_embedding['tsne'] = tsne.astype('float32')
            g_embedding['tsne'].attrs['embedding'] = 'spectral'
            g_embedding['tsne'].attrs['index'] = np.array([0, 1])

            g_embedding['umap'] = umap.astype('float32')
            g_embedding['umap'].attrs['embedding'] = 'spectral'
            g_embedding['umap'].attrs['index'] = np.array([0, 1])

            g_cluster = g_analysis.require_group('cluster')
            for n_cluster, clustering in zip(n_clusters, kmeans):
                s_cluster = 'kmeans-{:02d}'.format(n_cluster)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['k'] = n_cluster
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            for n_cluster, clustering in zip(n_clusters, dbscan):
                s_cluster = 'dbscan-eps={:.1f}'.format(n_cluster / 10.)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            s_cluster = 'hdbscan'
            g_cluster[s_cluster] = hdbscan
            g_cluster[s_cluster].attrs['embedding'] = 'spectral'
            g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            for n_cluster, clustering in zip(n_clusters, agglo):
                s_cluster = 'agglomerative-{:02d}'.format(n_cluster)
                g_cluster[s_cluster] = clustering
                g_cluster[s_cluster].attrs['embedding'] = 'spectral'
                g_cluster[s_cluster].attrs['k'] = n_cluster
                g_cluster[s_cluster].attrs['index'] = np.arange(embedding.shape[1], dtype='uint32')

            if train_flag is not None:
                g_cluster['train_split'] = train_flag


if __name__ == '__main__':
    main()
