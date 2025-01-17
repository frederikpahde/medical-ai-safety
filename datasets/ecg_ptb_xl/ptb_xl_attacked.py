import numpy as np
import pandas as pd
import torch
from datasets.ecg_ptb_xl.ptb_xl import ptb_xl_augmentation, PtbXlDataset


def get_ptb_xl_attacked_dataset(data_paths,
                                attacked_classes,
                                p_artifact=0,
                                artifact_type="defective_lead",
                                  **kwargs):
    
    transform = None
    return PtbXlAttackedDataset(data_paths, transform=transform, augmentation=ptb_xl_augmentation,
                                task="subdiagnostic", attacked_classes=attacked_classes,
                                p_artifact=p_artifact, artifact_type=artifact_type,
                                **kwargs)


class PtbXlAttackedDataset(PtbXlDataset):
    
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
        super().__init__(data_paths, transform, augmentation, task)
        self.attacked_classes = attacked_classes
        self.p_artifact = p_artifact
        self.artifact_type = artifact_type
        self.artifact_kwargs = artifact_kwargs
        np.random.seed(0)
        self.artifact_labels = np.array(
            [np.array([self.labels[idx][cl] == 1.0 for cl in attacked_classes]).any() and
             np.random.rand() < self.p_artifact
             for idx in range(len(self))])
        print(f"Initiated dataset with {self.artifact_labels.sum()} artifact samples")
        self.artifact_ids = np.where(self.artifact_labels)[0]
        self.sample_ids_by_artifact = {"artificial": self.artifact_ids, artifact_type: self.artifact_ids}
        self.clean_sample_ids = [i for i in range(len(self)) if i not in self.artifact_ids]

    def insert_defective_lead(self, x, lead_ids=[0], seq_length=100):
        x_man = x.clone()
        for lead_id in lead_ids:
            x_man[lead_id][:seq_length] = x_man[lead_id][:seq_length].max() * 2
        return x_man
    
    def amplify_beat(self, x, i, lead_ids, peak_ids, factor):
        x_man = x.clone()
        for peak_id in peak_ids:
            if len(self.metadata.iloc[i].r_peaks) >= (peak_id + 2):
                idxs_peak = (
                    int(np.array(self.metadata.iloc[i].r_peaks[peak_id-1:peak_id+1]).mean()),
                    int(np.array(self.metadata.iloc[i].r_peaks[peak_id:peak_id+2]).mean())

                )
                i1, i2 = idxs_peak
                for lead_id in lead_ids:
                    x_man[lead_id][i1:i2] = self.amplify(x_man[lead_id][i1:i2], factor)

        return x_man

    def amplify(self, values, factor):
        c = len(values) // 2
        factor_inc = np.linspace(1, factor, c)
        num_decr = c if c * 2 == len(values) else c +1
        factor_decr = np.linspace(factor, 1, num_decr)
        factors = np.concatenate([factor_inc, factor_decr])
        return values * factors

    def insert_artifact(self, x, i):
        if self.artifact_type == "defective_lead":
            assert "lead_ids" in self.artifact_kwargs
            assert "seq_length" in self.artifact_kwargs
            return self.insert_defective_lead(x, 
                                              self.artifact_kwargs["lead_ids"], 
                                              self.artifact_kwargs["seq_length"])
        elif self.artifact_type == "amplified_beat":
            return self.amplify_beat(x, i, 
                                     self.artifact_kwargs["lead_ids"], 
                                     self.artifact_kwargs["peak_ids"], 
                                     self.artifact_kwargs["amplify_factor"])
        else:
            raise ValueError(f"Unknown artifact: {self.artifact_type}")
        

    def __getitem__(self, index):
        x, y = self.data[index].T, self.labels[index]
        if self.artifact_labels[index]:
            x = self.insert_artifact(x, index)
        if self.transform:
            x = self.transform(x)
        if self.is_single_label:
            y = int(y)
        return x, y
    
    def make_single_label(self):
        print("Turn multi-label data into single-label")
        length_before = len(self)
        num_classes = len(self.classes)
        all_ds = []
        for cl in range(num_classes):
            idxs_class = np.where(self.labels[:, cl] == 1)[0]
            if cl not in self.attacked_classes:
                idxs_class = np.array([idx for idx in idxs_class if idx not in self.artifact_ids])
            ds_cl = self.get_subset_by_idxs(idxs_class)
            ds_cl.labels = np.ones(len(ds_cl)) * cl
            all_ds.append(ds_cl)

        # join
        self.metadata = pd.concat([ds.metadata for ds in all_ds]).copy()
        self.labels = np.concatenate([ds.labels for ds in all_ds]).copy().astype(np.uint8)
        self.data = torch.cat([ds.data for ds in all_ds]).clone()

        # update artifact labels
        self.artifact_labels = np.concatenate([ds.artifact_labels for ds in all_ds]).copy()
        self.artifact_ids = np.where(self.artifact_labels)[0]
        self.sample_ids_by_artifact = {"artificial": self.artifact_ids}
        self.clean_sample_ids = [i for i in range(len(self)) if i not in self.artifact_ids]

        del all_ds
        self.is_single_label = True
        print(f"Changed size from {length_before} to {len(self)}")
    
    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.artifact_labels = self.artifact_labels[np.array(idxs)]

        subset.artifact_ids = np.where(subset.artifact_labels)[0]
        subset.sample_ids_by_artifact = {"artificial": subset.artifact_ids}
        subset.clean_sample_ids = [i for i in range(len(subset)) if i not in subset.artifact_ids]
        return subset
