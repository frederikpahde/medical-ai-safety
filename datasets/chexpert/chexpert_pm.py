from pathlib import Path
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from datasets.base_dataset import BaseDataset

chexpert_augmentation = T.Compose([
    T.RandomHorizontalFlip(p=.25),
    T.RandomVerticalFlip(p=.25),
    T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=.25),
    T.RandomApply(transforms=[T.RandomRotation(degrees=(0, 30))], p=.25),
    T.RandomApply(transforms=[T.ColorJitter(brightness=.1, saturation=.1, hue=.1)], p=.25)
])

NORM_PARAMS_CHEXPERT = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
FN_NORMALIZE_CHEXPERT = T.Normalize(*NORM_PARAMS_CHEXPERT)

def get_chexpert_pm_dataset(data_paths,
                     normalize_data=True,
                     image_size=224,
                     binary_target=None,
                     artifact_ids_file=None,
                     **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(FN_NORMALIZE_CHEXPERT)

    transform = T.Compose(fns_transform)

    return CheXpertPMDataset(data_paths, transform=transform, binary_target=binary_target,
                           augmentation=chexpert_augmentation, artifact_ids_file=artifact_ids_file)


class CheXpertPMDataset(BaseDataset):
    classes = ["No Disease", "Disease"]
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 binary_target=None,
                 augmentation=None,
                 artifact_ids_file=None
                 ):
        super().__init__(data_paths, transform, augmentation, artifact_ids_file)

        self.base_path = data_paths[0]

        metadata = pd.read_csv(f"{self.base_path}/train.csv")

        path_metadata_pm = "data/che_pm_shortcut_labels.csv"

        path_metadata_pm = path_metadata_pm if os.path.isfile(path_metadata_pm) else f"../{path_metadata_pm}"
        path_metadata_pm = path_metadata_pm if os.path.isfile(path_metadata_pm) else f"../{path_metadata_pm}"
        assert os.path.isfile(path_metadata_pm), f"{path_metadata_pm} does not exist"

        metadata_pm = pd.read_csv(path_metadata_pm)
        metadata["path_pm_metadata_style"] = ["_".join(row.Path.split("/")[-3:]) for i, row in metadata.iterrows()]
        self.metadata = metadata.set_index("path_pm_metadata_style").join(metadata_pm.set_index("img_name")[["PM"]], how="inner").reset_index()

        self.normalize_fn = FN_NORMALIZE_CHEXPERT
        if binary_target is not None:
            assert binary_target in ["No Finding", "Cardiomegaly"] + self.classes, f"Binary target must be either 'No Finding' or one of {', '.join(self.classes)}, is: {binary_target}"
            abbr = {"Cardiomegaly": "CM"}
            self.classes = [f"Not '{abbr[binary_target]}'", abbr[binary_target]]

        self.binary_target = binary_target

        # Split train data into train/val by patient id
        val_split, test_split = .1, .1
        rng = np.random.default_rng(0)
        self.metadata["patient_id"] = self.metadata["Path"].str.split("/").str[2]
        all_patients = self.metadata["patient_id"].drop_duplicates()

        test_patients = rng.choice(all_patients, size=int(np.round(len(all_patients) * test_split)), replace=False)
        patients_left = sorted(np.array(list(set(all_patients) - set(test_patients))))
        val_patients = rng.choice(patients_left, size=int(np.round(len(all_patients) * val_split)), replace=False)

        idxs_all = np.arange(len(self.metadata))
        self.idxs_val = [i for i, patient_id in enumerate(self.metadata["patient_id"].values) if patient_id in val_patients]
        self.idxs_test = [i for i, patient_id in enumerate(self.metadata["patient_id"].values) if patient_id in test_patients]
        self.idxs_train = sorted(np.array(list(set(idxs_all) - set(self.idxs_val)- set(self.idxs_test))))

        
        if self.binary_target is None:
            dist = self.metadata[self.classes].fillna(0).replace(-1, 0).agg(sum).values
        else:
            dist = np.array([
                (self.metadata[self.binary_target] != 1).sum(),
                (self.metadata[self.binary_target] == 1).sum()
            ])
        self.weights = self.compute_weights(dist)

        self.sample_ids_by_artifact = {"pm": np.where(self.metadata["PM"].values == 1)[0]}

        self.all_artifact_sample_ids = [sample_id for _, sample_ids in self.sample_ids_by_artifact.items() for sample_id
                                        in sample_ids]
        self.clean_sample_ids = list(set(np.arange(len(self))) - set(self.all_artifact_sample_ids))

        print(f"Loaded CheXpert data with train/val/test: {len(self.idxs_train)}/{len(self.idxs_val)}/{len(self.idxs_test)}")

    def get_all_ids(self):
        return np.arange(len(self))

    def get_sample_name(self, i):
        return self.metadata.iloc[i]["Path"]

    def get_target(self, i):
        if self.binary_target is not None:
            target = torch.tensor((self.metadata.iloc[i][self.binary_target] == 1.0).astype(int))
        else:
            target = torch.tensor([self.metadata.iloc[i][cl] == 1 for cl in self.classes]).float()
        return target
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, i):

        row = self.metadata.iloc[i]

        img = Image.open(self.base_path / Path(*Path(row["Path"]).parts[1:])).convert("RGB")

        img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        target = self.get_target(i) 

        return img, target

    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        return subset
