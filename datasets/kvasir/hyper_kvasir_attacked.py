from pathlib import Path
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets.kvasir.hyper_kvasir import FN_NORMALIZE_KVASIR, HyperKvasirDataset, kvasir_augmentation
from utils.artificial_artifact import insert_artifact

def get_hyper_kvasir_attacked_dataset(data_paths,
                                      normalize_data=True,
                                      image_size=224,
                                      attacked_classes=[], p_artifact=.5, artifact_type='ch_text',
                                      **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(FN_NORMALIZE_KVASIR)

    transform = T.Compose(fns_transform)

    return HyperKvasirAttackedDataset(data_paths, transform=transform, augmentation=kvasir_augmentation, 
                                      attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type,
                                      image_size=image_size, **kwargs)


class HyperKvasirAttackedDataset(HyperKvasirDataset):
    def __init__(self,
                 data_paths,
                 transform=None,
                 augmentation=None,
                 attacked_classes=[],
                 image_size=224,
                 p_artifact=.5,
                 artifact_type="ch_text",
                 **artifact_kwargs
                 ):
        super().__init__(data_paths, transform, augmentation, None)


        self.attacked_classes = attacked_classes
        self.p_artifact = p_artifact
        self.image_size = image_size
        self.artifact_type = artifact_type
        self.transform_resize = T.Resize((image_size, image_size))
        self.artifact_kwargs = artifact_kwargs
        self.p_backdoor = artifact_kwargs.get('p_backdoor', 0)

        np.random.seed(0)

        self.artifact_labels = np.array(
            [(np.array([self.get_target(idx) in self.attacked_classes]) == 1.0).any() and
             np.random.rand() < self.p_artifact
             for idx in range(len(self))])
        self.artifact_ids = np.where(self.artifact_labels)[0]
        self.sample_ids_by_artifact = {"artificial": self.artifact_ids, artifact_type: self.artifact_ids}
        self.clean_sample_ids = [i for i in range(len(self)) if i not in self.artifact_ids]

    def add_artifact(self, img, idx):
        random.seed(idx)
        torch.manual_seed(idx)
        np.random.seed(idx)

        return insert_artifact(img, self.artifact_type, **self.artifact_kwargs)
    
    def __getitem__(self, i):

        class_idx = self.get_target(i)
        path = self.get_path(i)

        img = Image.open(self.base_path / Path(path))
        img = self.transform_resize(img)

        insert_backdoor = (np.random.rand() < self.p_backdoor) and (len(self.attacked_classes) > 0)

        if self.artifact_labels[i] or insert_backdoor:
            img, _ = self.add_artifact(img, i)

        if self.transform:
            img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        target = torch.tensor(class_idx).long() 
        if insert_backdoor:
            target = torch.tensor(self.attacked_classes[0])

        return img, target

    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.artifact_labels = self.artifact_labels[np.array(idxs)]
        subset.artifact_ids = np.where(subset.artifact_labels)[0]
        subset.sample_ids_by_artifact = {"artificial": subset.artifact_ids}
        subset.clean_sample_ids = [i for i in range(len(subset)) if i not in subset.artifact_ids]
        return subset
