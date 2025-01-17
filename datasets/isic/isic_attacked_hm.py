import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import glob
import torchvision.transforms as T
from datasets.isic.isic import isic_augmentation
from datasets.isic.isic_attacked import ISICAttackedDataset

def get_isic_attacked_hm_dataset(data_paths, 
                                 normalize_data=True, 
                                 binary_target=False, 
                                 image_size=224, 
                                 p_artifact=None,
                                 attacked_classes=None,
                                 artifact_type="ch_text", 
                                 **kwargs):

    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = T.Compose(fns_transform)
    
    return ISICAttackedHmDataset(data_paths, train=True, transform=transform, augmentation=isic_augmentation,
                         binary_target=binary_target, p_artifact=p_artifact, attacked_classes=attacked_classes,
                         artifact_type=artifact_type, image_size=image_size, **kwargs)



class ISICAttackedHmDataset(ISICAttackedDataset):
    def __init__(self, 
                 data_paths, 
                 train=False, 
                 transform=None, 
                 augmentation=None,
                 binary_target=False, 
                 attacked_classes=[], 
                 p_artifact=.5,
                 artifact_type='ch_text',
                 image_size=224,
                 **artifact_kwargs
                 ):
        super().__init__(data_paths, train, transform, augmentation, binary_target, 
                         attacked_classes, p_artifact, artifact_type, image_size, **artifact_kwargs)
        
        self.source_masks = artifact_kwargs.get("source_masks", "gt")

        fname_mask_config = "data/localization_layers.json"
        with open(fname_mask_config) as f:
            mask_config = json.load(f)
            _, mask_layer = mask_config["isic_attacked"]["artificial"]

        paths_masks = {
            "hm": f"data/localizations/hm/isic_attacked/{mask_layer}-svm/artificial",
            "bin": f"data/localizations/binary/isic_attacked/{mask_layer}-svm/artificial"
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
        row = self.metadata.loc[i]

        path = self.train_dirs_by_version[row.version] if self.train else self.test_dirs_by_version[row.version]
        img = Image.open(path / Path(row['image'] + '.jpg'))
        img = self.transform_resize(img)
        if self.artifact_labels[i]:
            img, mask = self.add_artifact(img, i)
            mask = mask.float()
            ## if required: overwrite mask with predicted one
            if self.source_masks in ("hm", "bin"):
                mask = torch.tensor(np.array(Image.open(self.metadata.iloc[i].hm_path))).clamp(min=0).float()
                mask /= mask.max()
        else:
            mask = torch.zeros((self.image_size, self.image_size)).float()

        img = self.transform(img)
        columns = self.metadata.columns.to_list()
        target = torch.Tensor([columns.index(row[row == 1.0].index[0]) - 1 if self.train else 0]).long()[0]
        return img, target, mask

    