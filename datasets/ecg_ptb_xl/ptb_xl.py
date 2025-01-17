import numpy as np
import pandas as pd
import torch
from datasets.base_dataset import BaseDataset


ptb_xl_augmentation = None

def get_ptb_xl_dataset(data_paths,
                       **kwargs):
    
    transform = None
    return PtbXlDataset(data_paths, transform=transform, 
                        augmentation=ptb_xl_augmentation,
                        task="subdiagnostic")


class PtbXlDataset(BaseDataset):

    classes = np.array([
        'NORM',    
        'LVH','RVH','SEHYP','LAO/LAE','RAO/RAE',
        'CLBBB','CRBBB','ILBBB','IRBBB','IVCD','LAFB/LPFB','WPW','_AVB',
        'AMI','IMI','LMI','PMI',   
        'ISCA','ISCI','ISC_','NST_','STTC'
        ])
    
    def __init__(self,
                 data_paths,
                 transform=None,
                 augmentation=None,
                 task="subdiagnostic"
                 ):
        super().__init__(data_paths, transform, augmentation, None)

        self.data, self.metadata, self.labels = self.load_data(data_paths[0], task)
        self.normalize_fn = None

        train_strats = [1,2,3,4,5,6,7,8]
        val_strats = [9]
        test_strats = [10]

        self.idxs_train = [i for i, fold in enumerate(self.metadata.strat_fold.values) if fold in train_strats]
        self.idxs_val = [i for i, fold in enumerate(self.metadata.strat_fold.values) if fold in val_strats]
        self.idxs_test = [i for i, fold in enumerate(self.metadata.strat_fold.values) if fold in test_strats]

        self.weights = self.compute_weights(self.labels.sum(0))
        self.sample_ids_by_artifact = {}
        self.clean_sample_ids = [i for i in range(len(self))]
        self.is_single_label = False

    def load_data(self, data_path, task):
        data = torch.from_numpy(np.load(f"{data_path}/raw100.pkl", allow_pickle=True)).float()
        metadata = pd.read_csv(f"{data_path}/ptbxl_database_enriched.csv",index_col=0)
        metadata.r_peaks = metadata.r_peaks.apply(lambda x: eval(x.replace('[  ','[').replace('[ ','[').replace('  ', ' ').replace(' ', ',')))
        metadata["my_id"] = np.arange(len(metadata))
        metadata = metadata.set_index("my_id")
        labels = torch.from_numpy(np.load(f"{data_path}/multihot_{task}.npy", allow_pickle=True)).float()
        return data, metadata, labels

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        if self.transform:
            return self.transform((self.data[index].T, self.labels[index]))

        label = self.labels[index]
        if self.is_single_label:
            label = int(label)

        return self.data[index].T, label


    def get_subset_by_idxs(self, idxs):
        subset = super().get_subset_by_idxs(idxs)
        subset.metadata = subset.metadata.iloc[idxs].reset_index(drop=True)
        subset.labels = subset.labels[idxs]
        subset.data = subset.data[idxs]
        
        return subset
    
    def get_target(self, i):
        return self.labels[i]
    
    def get_sample_name(self, i):
        return f"{i}"
    
    def make_single_label(self):
        print("Turn multi-label data into single-label")
        length_before = len(self)
        num_classes = len(self.classes)
        all_ds = []
        for cl in range(num_classes):
            idxs_class = np.where(self.labels[:, cl] == 1)[0]
            ds_cl = self.get_subset_by_idxs(idxs_class)
            ds_cl.labels = np.ones(len(ds_cl)) * cl
            all_ds.append(ds_cl)

        # join
        self.metadata = pd.concat([ds.metadata for ds in all_ds]).copy()
        self.labels = np.concatenate([ds.labels for ds in all_ds]).copy().astype(np.uint8)
        self.data = torch.cat([ds.data for ds in all_ds]).clone()
        del all_ds
        self.is_single_label = True
        print(f"Changed size from {length_before} to {len(self)}")
        