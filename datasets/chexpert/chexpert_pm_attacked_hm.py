from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from datasets.chexpert.chexpert_pm_attacked import CheXpertPMAttackedDataset, chexpert_augmentation, FN_NORMALIZE_CHEXPERT

def get_chexpert_pm_attacked_hm_dataset(data_paths,
                                    normalize_data=True,
                                    image_size=224,
                                    binary_target=None,
                                    attacked_classes=[], p_artifact=.5, artifact_type='ch_text',
                                    **kwargs):
    fns_transform = [
        T.Resize((image_size, image_size), interpolation=T.functional.InterpolationMode.BICUBIC),
        T.ToTensor()
    ]

    if normalize_data:
        fns_transform.append(FN_NORMALIZE_CHEXPERT)

    transform = T.Compose(fns_transform)

    return CheXpertPMAttackedHmDataset(data_paths, transform=transform, binary_target=binary_target,
                           augmentation=chexpert_augmentation,attacked_classes=attacked_classes, 
                           p_artifact=p_artifact, artifact_type=artifact_type, **kwargs)


class CheXpertPMAttackedHmDataset(CheXpertPMAttackedDataset):
    classes = ["No Disease", "Disease"]
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 binary_target=None,
                 augmentation=None,
                 attacked_classes=[],
                 image_size=224,
                 p_artifact=.5,
                 artifact_type="ch_text",
                 **artifact_kwargs
                 ):
        super().__init__(data_paths, transform, binary_target, augmentation, attacked_classes,
                         image_size, p_artifact, artifact_type, **artifact_kwargs)

    
    def __getitem__(self, i):

        row = self.metadata.iloc[i]

        img = Image.open(self.base_path / Path(*Path(row["Path"]).parts[1:])).convert("RGB")

        img = self.transform_resize(img)

        insert_backdoor = (np.random.rand() < self.p_backdoor) and (len(self.attacked_classes) > 0)

        if self.artifact_labels[i] or insert_backdoor:
            img, mask = self.add_artifact(img, i)
        else:
            mask = torch.zeros((self.image_size, self.image_size))

        if self.transform:
            img = self.transform(img)
        if self.do_augmentation:
            img = self.augmentation(img)

        target = self.get_target(i)
        if insert_backdoor:
            target = torch.tensor(self.attacked_classes[0])

        return img, target, mask.type(torch.uint8)
