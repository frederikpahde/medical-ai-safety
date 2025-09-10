import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import copy

class AttackedDataset(Dataset):
    
    def __init__(self, base_dataset, artifact_name):
        
        self.base_dataset = base_dataset
        art_samples = base_dataset.sample_ids_by_artifact[artifact_name]
        
        ## Collect artifacts
        gaussian = torchvision.transforms.GaussianBlur(kernel_size=41, sigma=5.0)
        self.artifacts = torch.stack([self.base_dataset[j][0] for j in art_samples], dim=0)
        self.masks = torch.stack([self.base_dataset[j][2] for j in art_samples])
        self.masks = gaussian(self.masks.clamp(min=0)) ** 1.0
        self.masks = self.masks / self.masks.abs().flatten(start_dim=1).max(1)[0][:, None, None]
        
        print(f"Collected {len(self.masks)} artifact samples")
        
        self.rng = np.random.default_rng(0)
        self.classes = base_dataset.classes
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, i):
        x, y, _ = self.base_dataset[i]
        
        pick = self.rng.choice(np.arange(len(self.masks)))
        m = self.masks[pick][None, :, :]
        artifact = self.artifacts[pick]
        x_att = x * (1 - m) + artifact * m
        
        return x_att, y, m
    
    def reverse_normalization(self, data):
        return self.base_dataset.reverse_normalization(data)
    
    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.base_dataset = subset.base_dataset.get_subset_by_idxs(idxs)
        return subset