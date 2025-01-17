import numpy as np
import pandas as pd
import torch
from datasets.ecg_ptb_xl.ptb_xl_attacked import ptb_xl_augmentation, PtbXlAttackedDataset


def get_ptb_xl_attacked_hm_dataset(data_paths,
                                   attacked_classes,
                                   p_artifact=0,
                                   artifact_type="defective_lead",
                                   **kwargs):
    
    transform = None
    return PtbXlAttackedHmDataset(data_paths, transform=transform, augmentation=ptb_xl_augmentation,
                                task="subdiagnostic", attacked_classes=attacked_classes,
                                p_artifact=p_artifact, artifact_type=artifact_type,
                                **kwargs)


class PtbXlAttackedHmDataset(PtbXlAttackedDataset):
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 augmentation=None,
                 task="subdiagnostic",
                 attacked_classes=[],
                 p_artifact=.2,
                 artifact_type="defective_lead",
                 **artifact_kwargs
                 ):
        super().__init__(data_paths, transform, augmentation, task,
                         attacked_classes, p_artifact, artifact_type,
                         **artifact_kwargs)
        
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        assert self.artifact_type == "defective_lead", f"artifact type '{self.artifact_type}' not supported"
        mask = torch.zeros_like(x)
        if self.artifact_labels[i]:
            mask[:,:self.artifact_kwargs["seq_length"]] = 1
        return x, y, mask.type(torch.uint8)[0]