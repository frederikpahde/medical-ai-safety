import torch
import os
from sklearn.decomposition import NMF
import warnings
import numpy as np

class NMFModule(torch.nn.Module):
    def __init__(self, H, device="cuda", pool="max"):
        super().__init__()
        self.H = H
        
        if pool == "max":
            self.pool = torch.nn.AdaptiveMaxPool2d(1)
        elif pool == "avg":
            self.pool = torch.nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pooling: {pool}")
            
        self.project = torch.nn.Linear(*H.T.shape, bias=False)
        self.project.weight.data = H
        self.to(device)
        
    def forward(self, x):
        nmf_emb = self.project(self.pool(x).squeeze(-1).squeeze(-1))
        return x

def run_nmf(n_concepts, config, scope, mode):

    artifact_type = config.get('artifact_type', None)
    artifact_extension = f"_{artifact_type}-{config['p_artifact']}" if artifact_type else ""

    scope_str = "all" if scope is None else "-".join([str(s) for s in scope])
    fname_nmf_params = f"{config['dir_precomputed_data']}/nmf/{config['dataset_name']}{artifact_extension}/{config['model_name']}/H_{config['layer_name']}_{n_concepts}_{scope_str}.pth"

    if os.path.isfile(fname_nmf_params):
        print(f"Loading existing NMF weights from {fname_nmf_params}")
        H = torch.load(fname_nmf_params)
    else:
        print(f"Run NMF")
        path_precomputed_data = f"{config['dir_precomputed_data']}/global_relevances_and_activations/{config['dataset_name']}{artifact_extension}/{config['model_name']}"
        ## Load pre-computed activations
        vecs = []
        sample_ids = []
        for class_id in scope:
            path_precomputed_activations = f"{path_precomputed_data}/{config['layer_name']}_class_{class_id}_all.pth"
            print(f"reading precomputed relevances/activations from {path_precomputed_activations}")
            data = torch.load(path_precomputed_activations)
            if data['samples']:
                sample_ids += data['samples']
                vecs.append(torch.stack(data[mode], 0))

        vecs = torch.cat(vecs, 0)
        sample_ids = np.array(sample_ids)

        if (vecs < 0).any():
            warnings.warn(f"Activations below 0 are clamped for NMF computation")
            vecs = vecs.clamp(min=0)

        nmf = NMF(n_components=n_concepts, init='random', random_state=0)
        W = nmf.fit_transform(vecs)
        H = nmf.components_
        H = torch.from_numpy(H)
        os.makedirs(os.path.dirname(fname_nmf_params), exist_ok=True)
        torch.save(H, fname_nmf_params)

    return H.type(torch.float32)

def overwrite_layer_nmf(model, layer_name, H, device="cuda", pooling="max"):
    modules = layer_name.split('.')
    sub_module = model
    for module in modules[:-1]:
        sub_module = getattr(sub_module, module)
    l = getattr(sub_module, modules[-1])
    l_new = torch.nn.Sequential(l, NMFModule(H, device, pooling))
    setattr(sub_module, modules[-1], l_new)