import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from datasets.base_dataset import BaseDataset


isic_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)
])

NORM_PARAMS_ISIC24 = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

def get_isic24_dataset(data_paths,
                       normalize_data=True,
                       image_size=224,
                       artifact_ids_file=None,
                     **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize(*NORM_PARAMS_ISIC24))

    transform = T.Compose(fns_transform)

    return ISIC24Dataset(data_paths, transform=transform, augmentation=isic_augmentation,
                         artifact_ids_file=artifact_ids_file)



class ISIC24Dataset(BaseDataset):
    classes = [0,1]
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 augmentation=None,
                 artifact_ids_file=None
                 ):
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)
        
        self.normalize_fn = T.Normalize(*NORM_PARAMS_ISIC24)

        self.path = data_paths[0]
        self.metadata = pd.read_csv(f"{self.path}/train-metadata.csv")

        ## only use subset of benign samples
        idxs_malignant = np.where(self.metadata["target"]==1)[0]
        idxs_bengin = np.where(self.metadata["target"]==0)[0]
        rng = np.random.default_rng(0)
        idxs_benign_subset = rng.choice(idxs_bengin, size=10000)
        idxs_use = sorted(np.concatenate([idxs_malignant, idxs_benign_subset]))
        self.metadata = self.metadata.iloc[idxs_use].reset_index()

        # compute class balance
        dist = np.array([len(idxs_benign_subset), len(idxs_malignant)])
        self.weights = self.compute_weights(dist)

        # split into train/val/test by patient ID
        val_split, test_split = .05, .1
        rng = np.random.default_rng(0)
        all_patients = self.metadata["patient_id"].drop_duplicates()
        test_patients = rng.choice(all_patients, size=int(np.round(len(all_patients) * test_split)), replace=False)
        patients_left = sorted(np.array(list(set(all_patients) - set(test_patients))))
        rng = np.random.default_rng(0)
        val_patients = rng.choice(patients_left, size=int(np.round(len(all_patients) * val_split)), replace=False)

        idxs_all = np.arange(len(self.metadata))
        self.idxs_val = [i for i, patient_id in enumerate(self.metadata["patient_id"].values) if patient_id in val_patients]
        self.idxs_test = [i for i, patient_id in enumerate(self.metadata["patient_id"].values) if patient_id in test_patients]
        self.idxs_train = sorted(np.array(list(set(idxs_all) - set(self.idxs_val)- set(self.idxs_test))))

        self.sample_ids_by_artifact = self.get_sample_ids_by_artifact()

        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

        num_mel_train = self.metadata.iloc[self.idxs_train]['target'].sum()
        num_mel_val = self.metadata.iloc[self.idxs_val]['target'].sum()
        num_mel_test = self.metadata.iloc[self.idxs_test]['target'].sum()
        print(f"Loaded ISIC24 data with train/val/test: {len(self.idxs_train)} (MEL: {num_mel_train})/{len(self.idxs_val)} (MEL: {num_mel_val})/{len(self.idxs_test)} (MEL: {num_mel_test})")

    def get_all_ids(self):
        return list(self.metadata.isic_id.values)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):

        row = self.metadata.loc[i]

        img_path = f"{self.path}/train-image/image/{row['isic_id']}.jpg"
        img = Image.open(img_path)

        img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        target = self.get_target(i)

        return img, target

    def get_sample_name(self, i):
        return self.metadata.loc[i]["isic_id"]

    def get_target(self, i):
        target = torch.Tensor([self.metadata.loc[i]['target']]).long()[0]
        return target

    def get_target_name(self, i):
        return self.get_target(i)
    
    def get_class_id_by_name(self, class_name):
        return class_name
    
    def get_subset_by_idxs(self, idxs):
        subset = copy.deepcopy(self)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset
