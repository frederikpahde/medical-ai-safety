from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from collections import Counter
from datasets.base_dataset import BaseDataset

DISEASE = [
    "barretts",
    "short-segment-barretts",
    "oesophagitis-a",
    "oesophagitis-b-d",
    "hemorrhoids",
    "hemorroids",
    "polyp",
    "ulcerative-colitis-grade-0-1",
    "ulcerative-colitis-grade-1-2",
    "ulcerative-colitis-grade-2-3",
    "ulcerative-colitis-grade-1",
    "ulcerative-colitis-grade-2",
    "ulcerative-colitis-grade-3",
    "dyed-lifted-polyps",
    "impacted-stool",
]

NON_DISEASE = [
    "bbps-0-1",
    "bbps-2-3",
    "dyed-resection-margins",
    "ileum",
    "retroflex-rectum",
    "retroflex-stomach",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
]

kvasir_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)
])

NORM_PARAMS_KVASIR = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
FN_NORMALIZE_KVASIR = T.Normalize(*NORM_PARAMS_KVASIR)

def get_hyper_kvasir_dataset(data_paths,
                     normalize_data=True,
                     image_size=224,
                     artifact_ids_file=None,
                     **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(FN_NORMALIZE_KVASIR)

    transform = T.Compose(fns_transform)

    return HyperKvasirDataset(data_paths, transform=transform, augmentation=kvasir_augmentation, 
                       artifact_ids_file=artifact_ids_file)


class HyperKvasirDataset(BaseDataset):
    classes = ["Non-Disease", "Disease"]
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 augmentation=None,
                 artifact_ids_file=None
                 ):
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)

        self.base_path = data_paths[0]
        
        self.metadata = pd.read_csv(f"{self.base_path}/image-labels.csv")

        self.disease = DISEASE
        
        self.non_disease = NON_DISEASE
        
        self.metadata[self.metadata["Finding"].isin(self.disease + self.non_disease)].reset_index()
        self.normalize_fn = FN_NORMALIZE_KVASIR
        
        labels = [self.get_target(i) for i in range(len(self))]
        sorted_counts = sorted(Counter(labels).items())
        dist = torch.tensor([count for _, count in sorted_counts])
        self.weights = self.compute_weights(dist)

        

        self.idxs_train, self.idxs_val, self.idxs_test = self.do_train_val_test_split(.1, .1)
        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()

        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

        self.str_tract_dict = {
            "Lower GI": "lower-gi-tract",
            "Upper GI": "upper-gi-tract",
        }

        print(f"Loaded data ({len(self)})")


    def get_path(self, i):
        row = self.metadata.iloc[i]
        fpath = f"{self.base_path}/{self.str_tract_dict[row['Organ']]}/{row['Classification']}/{row['Finding']}/{row['Video file']}.jpg"
        return fpath
    
    def get_all_ids(self):
        return range(len(self))

    def get_sample_name(self, i):
        return i

    def get_target(self, i):
        target = 1 if self.metadata.iloc[i]["Finding"] in self.disease else 0
        return target
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):

        class_idx = self.get_target(i)
        path = self.get_path(i)

        img = Image.open(self.base_path / Path(path))

        img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        target = torch.tensor(class_idx).long() 

        return img, target

    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset
