import json
import numpy as np
import torch
import glob
from PIL import Image
import torchvision.transforms as T
from datasets.kvasir.hyper_kvasir import HyperKvasirDataset, kvasir_augmentation

def get_hyper_kvasir_hm_dataset(data_paths, 
                        normalize_data=True, 
                        image_size=224, 
                        artifact_ids_file=None,
                        artifact=None,
                        **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return HyperKvasirHmDataset(data_paths, transform=transform, augmentation=kvasir_augmentation, 
                         artifact_ids_file=artifact_ids_file, artifact=artifact)



class HyperKvasirHmDataset(HyperKvasirDataset):
    def __init__(self, 
                 data_paths, 
                 transform=None, 
                 augmentation=None,
                 artifact_ids_file=None,
                 artifact=None
                 ):
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)
        
        fname_mask_config = "data/localization_layers.json"
        with open(fname_mask_config) as f:
            mask_config = json.load(f)["hyper_kvasir"]
            
        self.hm_path = f"data/localizations/hm/hyper_kvasir"
        artifacts = artifact.split("-")
        artifact_paths = []
        for artifact in artifacts:
            print("LOADING", artifact)
            artifact_paths += glob.glob(f"{self.hm_path}/{mask_config[artifact][1]}-svm/{artifact}/*")
        print(f"Localized artifacts: {len(artifact_paths)}")
        artifact_sample_ids = np.array([int(x.split("/")[-1].split(".")[0]) for x in artifact_paths])
        self.artifact_ids = artifact_sample_ids
        self.hms = ["" for _ in range(len(self.metadata))]
        for i, j in enumerate(artifact_sample_ids):
            path = artifact_paths[i]
            if self.hms[j]:
                self.hms[j] += f",{path}"
            else:
                self.hms[j] += f"{path}"

        self.metadata["hms"] = self.hms
        
    
    def __getitem__(self, i):

        img, target = super().__getitem__(i)
        if self.metadata["hms"].loc[i]:
            
            heatmaps = torch.stack(
                [torch.tensor(np.asarray(Image.open(hm))) for hm in self.metadata["hms"].loc[i].split(",")]).clamp(
                min=0)
            
            # sum heatmaps after normalizing each one
            heatmaps = heatmaps / heatmaps.flatten(start_dim=1).max(dim=1).values[:, None, None]
            heatmaps = heatmaps.sum(dim=0).float()
        else:
            heatmaps = torch.zeros_like(img[0])

        return img, target, heatmaps
