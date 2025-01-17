# code inspired from https://github.com/hhi-aml/xai4ecg/blob/master/code/utils.py

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

leads = np.array(['I','II','III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
label_mappings = {
    'diagnostic':['1AVB', '2AVB', '3AVB', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LPFB', 'LVH', 'NDT', 'NORM', 'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'WPW'],
    'subdiagnostic':['AMI', 'CLBBB', 'CRBBB', 'ILBBB', 'IMI', 'IRBBB', 'ISCA', 'ISCI', 'ISC_', 'IVCD', 'LAFB/LPFB', 'LAO/LAE', 'LMI', 'LVH', 'NORM', 'NST_', 'PMI', 'RAO/RAE', 'RVH', 'SEHYP', 'STTC', 'WPW', '_AVB'],
    'superdiagnostic':['CD', 'HYP', 'MI', 'NORM', 'STTC'],
    'rhythm':['AFIB', 'AFLT', 'BIGU', 'PACE', 'PSVT', 'SARRH', 'SBRAD', 'SR', 'STACH', 'SVARR', 'SVTAC', 'TRIGU'],
    'form':['ABQRS', 'DIG', 'HVOLT', 'INVT', 'LNGQT', 'LOWT', 'LPR', 'LVOLT', 'NDT', 'NST_', 'NT_', 'PAC', 'PRC(S)', 'PVC', 'QWAVE', 'STD_', 'STE_', 'TAB_', 'VCLVH'],
    'all':['1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI', 'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL', 'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN', 'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT', 'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE', 'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD', 'SEHYP', 'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU', 'VCLVH', 'WPW']
}
subdiag_ordering = np.array([
    'NORM',    
    'LVH','RVH','SEHYP','LAO/LAE','RAO/RAE',
    'CLBBB','CRBBB','ILBBB','IRBBB','IVCD','LAFB/LPFB','WPW','_AVB',
    'AMI','IMI','LMI','PMI',   
    'ISCA','ISCI','ISC_','NST_','STTC'
])

def predict_samples(model, signals, batch_size=128):
    predictions = []
    for i in tqdm(range(len(signals)//batch_size + 1)):
        yi = model.forward(torch.from_numpy(np.swapaxes(signals[i*batch_size:(i+1)*batch_size],1,2)).type(torch.float).to('cuda')).cpu().detach().numpy()
        predictions.append(yi)
    predictions = np.concatenate(predictions)
    return predictions

def get_dataset(data_path='../data/ptbxl/', sampling_rate=100):
    signals = np.load(data_path+'raw'+str(sampling_rate)+'.pkl', allow_pickle=True)
    dataset = pd.read_csv(data_path+'ptbxl_database_enriched.csv', index_col=0)
    dataset.r_peaks = dataset.r_peaks.apply(lambda x: eval(x.replace('[  ','[').replace('[ ','[').replace('  ', ' ').replace(' ', ',')))
    return signals, dataset

def get_dataset_label(task='subdiagnostic', data_path='../data/ptbxl/', sampling_rate=100):
    signals, dataset = get_dataset(data_path, sampling_rate)
    labels = np.load(data_path+'multihot_'+str(task)+'.npy', allow_pickle=True)
    return signals, dataset, labels

def get_beats(signals, df,n_from=0,n_to=1000, t_before=30, t_after=50, only_beats=True):
    beats = np.concatenate([[signals[i][ri-t_before:ri+t_after,:] for ri in np.array(df.iloc[i].r_peaks).astype(int)-n_from if (ri<len(signals[i])-t_after)&(ri > t_before)] for i in tqdm(range(len(signals))) if len([True for ri in np.array(df.iloc[i].r_peaks).astype(int)-n_from if (ri<len(signals[i])-t_after)&(ri > t_before)])>0])
    if only_beats:
        return beats
    else:
        ecg_ids = np.concatenate([[df.iloc[i].index]*len(df.iloc[i].r_peaks) for i in tqdm(range(len(signals)))])
        ids = np.concatenate([[i for ri in np.array(df.iloc[i].r_peaks).astype(int)-n_from if (ri<len(signals[i])-t_after)&(ri > t_before)] for i in tqdm(range(len(signals)))])
        return beats, ecg_ids, ids

def get_regression_data(sampling_rate=100,addon_path='addons/',data_path='data/ptbxl/', task='T_Wave_Amplitude', clip=False):

    df = pd.read_csv(addon_path+'unig_features.csv')
    df = df.sort_values('ecg_id').reset_index()

    cols = []
    for lead in leads:
        if task == 'T_Wave_Amplitude':
            df['T_amp_'+str(lead)] = df['T_Amp_'+str(lead)]
            cols.append('T_amp_'+str(lead))
        elif task == 'P_Wave_Amplitude':
            df['P_amp_'+str(lead)] = df['P_Amp_'+str(lead)]
            cols.append('P_amp_'+str(lead))
        elif task == 'R_Peak_Amplitude':
            df['R_amp_'+str(lead)] = df['R_Amp_'+str(lead)]
            cols.append('R_amp_'+str(lead))
        elif task == 'S_Peak_Amplitude':
            df['S_amp_'+str(lead)] = df['S_Amp_'+str(lead)]
            cols.append('S_amp_'+str(lead))
        elif task == 'Q_Peak_Amplitude':
            df['Q_amp_'+str(lead)] = df['Q_Amp_'+str(lead)]
            cols.append('Q_amp_'+str(lead))

    # clean qrs-complexes
    df = df[(df[cols].isna().values.sum(axis=-1) == 0)]

    # load dataset and dataframe
    signals = np.load(data_path+'raw'+str(sampling_rate)+'.pkl', allow_pickle=True)
    dataset = pd.read_csv(data_path+'ptbxl_database_enriched.csv', index_col=0)

    # merge data on shared ecg_ids
    shared_idxs = np.array(list(set(df.ecg_id).intersection(set(dataset.index))))
    mask = dataset.index.isin(shared_idxs)
    X = signals[mask]
    dataset = dataset[mask]
    df = df[df.ecg_id.isin(shared_idxs)]

    dataset.r_peaks = dataset.r_peaks.apply(lambda x: eval(x.replace('[  ','[').replace('[ ','[').replace('  ', ' ').replace(' ', ',')))
    

    if clip:
        vmax = np.quantile(df[cols].values.flatten(), q=.999)
        Y = df[cols].clip(-vmax,vmax).values
    else:
        Y = df[cols].values
    return X, Y, dataset