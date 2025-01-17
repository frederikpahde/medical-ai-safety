import json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import glob
from datasets.kvasir.hyper_kvasir_attacked import FN_NORMALIZE_KVASIR, HyperKvasirAttackedDataset, kvasir_augmentation

def get_hyper_kvasir_attacked_hm_dataset(data_paths,
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

    return HyperKvasirAttackedHmDataset(data_paths, transform=transform, augmentation=kvasir_augmentation, 
                                      attacked_classes=attacked_classes, p_artifact=p_artifact, artifact_type=artifact_type,
                                      image_size=image_size, **kwargs)


class HyperKvasirAttackedHmDataset(HyperKvasirAttackedDataset):
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
        super().__init__(data_paths, transform, augmentation, attacked_classes, image_size,
                         p_artifact, artifact_type, **artifact_kwargs)
        
        self.source_masks = artifact_kwargs.get("source_masks", "gt")

        fname_mask_config = "data/localization_layers.json"
        with open(fname_mask_config) as f:
            mask_config = json.load(f)["hyper_kvasir"]["artificial"]

        paths_masks = {
            "hm": f"data/localizations/hm/hyper_kvasir_attacked/{mask_config[1]}-svm/artificial",
            "bin": f"data/localizations/binary/hyper_kvasir_attacked/{mask_config[1]}-svm/artificial"
        }

        if self.source_masks in paths_masks.keys():
            self.mask_path = paths_masks[self.source_masks]
            
            print(f"Using masks from: {self.mask_path}")
            artifact_paths = glob.glob(f"{self.mask_path}/*")
            mask_sample_ids = np.array([int(x.split("/")[-1].split(".")[0]) for x in artifact_paths])
            assert (sorted(mask_sample_ids) == self.artifact_ids).all(), "Masks-sample-IDs and artifact-IDs do not match"
            hm_paths = [None] * len(self)
            for sid, path in zip(mask_sample_ids, artifact_paths):
                hm_paths[sid] = path 
            self.metadata["hm_path"] = hm_paths
    
    def __getitem__(self, i):

        class_idx = self.get_target(i)
        path = self.get_path(i)

        img = Image.open(self.base_path / Path(path))
        img = self.transform_resize(img)

        insert_backdoor = (np.random.rand() < self.p_backdoor) and (len(self.attacked_classes) > 0)

        if self.artifact_labels[i] or insert_backdoor:
            img, mask = self.add_artifact(img, i)
            mask = mask.float()

            ## if required: overwrite mask with predicted one
            if self.source_masks in ("hm", "bin"):
                mask = torch.tensor(np.array(Image.open(self.metadata.iloc[i].hm_path))).clamp(min=0).float()
                mask /= mask.max()
        else:
            mask = torch.zeros((self.image_size, self.image_size)).float()

        if self.transform:
            img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        target = torch.tensor(class_idx).long()
        if insert_backdoor:
            target = torch.tensor(self.attacked_classes[0])

        return img, target, mask

