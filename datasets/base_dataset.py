import json
import copy
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_paths, transform=None, augmentation=None, artifact_ids_file=None):
        self.data_paths = data_paths
        self.transform = transform
        self.augmentation = augmentation
        self.do_augmentation = False

        self.mean = torch.Tensor((0.5, 0.5, 0.5))
        self.var = torch.Tensor((0.5, 0.5, 0.5))

        print(f"Reading artifact IDs from {artifact_ids_file}")
        if artifact_ids_file:
            with open(artifact_ids_file, "r") as file:
                self.ids_by_artifact = json.load(file)
        else:
            self.ids_by_artifact = None

    def do_train_val_test_split(self, val_split=.1, test_split=.1):
        rng = np.random.default_rng(0)
        idxs_all = np.arange(len(self))
        idxs_val = np.array(sorted(rng.choice(idxs_all, size=int(np.round(len(idxs_all) * val_split)), replace=False)))
        idxs_left = np.array(list(set(idxs_all) - set(idxs_val)))
        idxs_test = np.array(
            sorted(rng.choice(idxs_left, size=int(np.round(len(idxs_all) * test_split)), replace=False)))
        idxs_train = np.array(sorted(list(set(idxs_left) - set(idxs_test))))

        return idxs_train, idxs_val, idxs_test

    def get_all_ids(self):
        raise NotImplementedError()

    def get_sample_name(self, i):
        raise NotImplementedError()

    def get_target(self, i):
        raise NotImplementedError()

    def get_target_name(self, i):
        return self.get_target(i)
    
    def get_class_id_by_name(self, class_name):
        return class_name
    
    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        idx_map = {idx_old: idx_new for idx_new, idx_old in enumerate(idxs)}
        subset.clean_sample_ids = [idx_map.get(i) for i in subset.clean_sample_ids if i in idx_map.keys()]
        subset.sample_ids_by_artifact = {artifact: [idx_map.get(i) for i in old_idxs if i in idx_map.keys()]
                                         for artifact, old_idxs in subset.sample_ids_by_artifact.items()}
        return subset
    
    def get_sample_ids_by_artifact(self):
        all_ids = self.get_all_ids()

        sample_ids_by_artifact = {
            artifact: [all_ids.index(artifact_id) for artifact_id in artifact_ids if artifact_id in all_ids]
            for artifact, artifact_ids in self.ids_by_artifact.items()
        } if self.ids_by_artifact is not None else {}

        return sample_ids_by_artifact

    def compute_weights(self, dist):
        return torch.tensor((dist > 0) / (dist + 1e-8) * dist.max()).float()

    def reverse_normalization(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone() + 0
        mean = self.mean.to(data)
        var = self.var.to(data)
        data *= var[:, None, None]
        data += mean[:, None, None]
        return torch.multiply(data, 255).type(torch.int16)
    