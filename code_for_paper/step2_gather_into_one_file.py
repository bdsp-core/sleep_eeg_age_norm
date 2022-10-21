"""
read all feature files and write into a single h5 file
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from tqdm import tqdm

# use "before" to indiate the missing sleep stages in ChicagoPediatric are not filled by model prediction
# use "after" to indiate the missing sleep stages in ChicagoPediatric are filled by model prediction
before_after_prediction = 'after'
if before_after_prediction not in ['before', 'after']:
    raise SystemExit('must be before/after, got %s'%before_after_prediction)
    
output_path = 'all_data_AHI15_Diag_betterartifact_apnea_removed.h5'

df = pd.read_excel('../mycode/all_data_paths.xlsx')

MGH_EEG_channels = np.array(['F3A2','F4A1','C3A2','C4A1','O1A2','O2A1'])
MGH_channel_id = [2,3,4,5]
MGH_EEG_channels = MGH_EEG_channels[MGH_channel_id]

CHIMES_EEG_channels = np.array(['C4A1','C3A2','O2A1','O1A2'])
CHIMES_channel_id = [1,0,3,2]
CHIMES_EEG_channels = CHIMES_EEG_channels[CHIMES_channel_id]

ChicagoPediatric_EEG_channels = np.array(['C4A1','C3A2','O2A1','O1A2'])
ChicagoPediatric_channel_id = [1,0,3,2]
ChicagoPediatric_EEG_channels = ChicagoPediatric_EEG_channels[ChicagoPediatric_channel_id]

assert np.all(MGH_EEG_channels==CHIMES_EEG_channels) and np.all(CHIMES_EEG_channels==ChicagoPediatric_EEG_channels)
EEG_channels = MGH_EEG_channels
nch = len(EEG_channels)
    
res1 = sio.loadmat(df.feature_file[df.Dataset=='MGH'].iloc[0], variable_names=['EEG_frequency'])
res2 = sio.loadmat(df.feature_file[df.Dataset=='CHIMES'].iloc[0], variable_names=['EEG_frequency'])
res3 = sio.loadmat(df.feature_file[df.Dataset=='ChicagoPediatric'].iloc[0], variable_names=['EEG_frequency'])
assert np.allclose(res1['EEG_frequency'], res2['EEG_frequency']) and np.allclose(res2['EEG_frequency'], res3['EEG_frequency'])
    
freq_downsample = 3
with h5py.File(output_path, 'w') as f:
    # create datasets
    dtypef = 'float16'
    dtypei = 'int32'
    dtypes = np.array([b'Feature_TwinData3_1000.mat               ']).dtype
    chunk_size = 128  # np.random.rand().astype(dtypef).nbytes/1024./1024./8. [MB]
    N = 0
    
    # get shapes
    spec_shape = [x[1][1:] for x in sio.whosmat(df.feature_file.iloc[0]) if x[0]=='EEG_specs'][0]
    spec_shape = (int(np.ceil(spec_shape[0]*1./freq_downsample)), nch)
    feature_shape = [x[1][1:] for x in sio.whosmat(df.feature_file[df.Dataset=='ChicagoPediatric'].iloc[0]) if x[0]=='EEG_features'][0]
    
    # create dataset
    dspec = f.create_dataset('spec', shape=(0,)+spec_shape, maxshape=(None,)+spec_shape,
                    chunks=(chunk_size,)+spec_shape, dtype=dtypef)
    dsleep_stage = f.create_dataset('sleep_stage', shape=(0,), maxshape=(None,), dtype=dtypef)
    dseg_time = f.create_dataset('seg_time', shape=(0,), maxshape=(None,), dtype=dtypef)
    dsubject = f.create_dataset('subject', shape=(0,), maxshape=(None,), dtype=dtypes)
    dage = f.create_dataset('age', shape=(0,), maxshape=(None,), dtype=dtypef)
    dsex = f.create_dataset('sex', shape=(0,), maxshape=(None,), dtype=dtypei)
    dfeature = f.create_dataset('feature', shape=(0,)+feature_shape, maxshape=(None,)+feature_shape,
                    chunks=(chunk_size,)+feature_shape, dtype=dtypef)
    
    
    for ii in tqdm(range(len(df))):
        dataset = df.Dataset.iloc[ii]
        this_path = df.feature_file.iloc[ii]
        if not os.path.exists(this_path):
            continue
        res = sio.loadmat(this_path)
        spec_ = res['EEG_specs']
        
        # downsample spectrum
        spec_ = spec_[:,::freq_downsample]
        
        if 'sleep_stages' in res:
            sleep_stage_ = res['sleep_stages'].flatten()
        else:
            if before_after_prediction=='before':
                sleep_stage_ = np.zeros(len(spec_)) + np.nan
            else:
                if 'predicted_sleep_stages_smoothed' in res:
                    sleep_stage_ = np.argmax(res['predicted_sleep_stages_smoothed'], axis=1)+1#predicted_sleep_stages
                else:
                    continue
        if 'EEG_features' in res:
            features_ = res['EEG_features']
        else:
            features_ = np.zeros((len(spec_),)+feature_shape) + np.nan
        seg_times_ = res['seg_times'].flatten()
        subject_ = np.array([os.path.basename(this_path)]*len(spec_))
        if dataset == 'CHIMES':
            age_ = np.array([res['age'][0,0]]*len(spec_))
            sex_ = np.array([-1]*len(spec_))
        else:
            age_ = df.age.iloc[ii]
            age_ = np.array([age_]*len(spec_))
            sex_ = df.sex.iloc[ii]
            sex_ = int(sex_=='M') if type(sex_)==str else -1
            sex_ = np.array([sex_]*len(spec_))
        
        # convert to db
        spec_ = 10*np.log10(spec_)
        
        # remove artifacts and apnea
        goodids = ((sleep_stage_==5)|(res['apnea_indicator'].flatten()==0))&(res['artifact_indicator'].flatten()==0)
        spec_ = spec_[goodids]
        sleep_stage_ = sleep_stage_[goodids]
        seg_times_ = seg_times_[goodids]
        subject_ = subject_[goodids]
        age_ = age_[goodids]
        sex_ = sex_[goodids]
        features_ = features_[goodids]
            
        # remove nan in spec
        notnanids = np.where(~np.any(np.isnan(spec_), axis=(1,2)))[0]
        if len(notnanids)<len(spec_):
            spec_ = spec_[notnanids]
            sleep_stage_ = sleep_stage_[notnanids]
            seg_times_ = seg_times_[notnanids]
            subject_ = subject_[notnanids]
            age_ = age_[notnanids]
            sex_ = sex_[notnanids]
            features_ = features_[notnanids]
        
        if dataset == 'MGH':
            spec_ = spec_[:,:,MGH_channel_id]
        elif dataset == 'CHIMES':
            spec_ = spec_[:,:,CHIMES_channel_id]
        elif dataset == 'ChicagoPediatric':
            spec_ = spec_[:,:,ChicagoPediatric_channel_id]
        
        if ii==0:
            Fs = res['Fs'][0,0]
            freq = res['EEG_frequency'].flatten()
            # downsample spectrum
            freq = freq[::freq_downsample]
    
            dfreq = f.create_dataset('freq', shape=freq.shape, dtype=dtypef)
            dFs = f.create_dataset('Fs', shape=(), dtype=dtypef)
            dchannelname = f.create_dataset('channelname', shape=(nch,), dtype=dtypes)
            
            dfreq[:] = freq
            dFs[()] = Fs
            dchannelname[:] = EEG_channels.astype(bytes)
    
        dspec.resize(N + len(spec_), axis=0); dspec[N:] = spec_
        dfeature.resize(N + len(features_), axis=0); dfeature[N:] = features_
        dsleep_stage.resize(N + len(sleep_stage_), axis=0); dsleep_stage[N:] = sleep_stage_
        dseg_time.resize(N + len(seg_times_), axis=0); dseg_time[N:] = seg_times_
        dsubject.resize(N + len(subject_), axis=0); dsubject[N:] = subject_.astype(bytes)
        dage.resize(N + len(age_), axis=0); dage[N:] = age_
        dsex.resize(N + len(sex_), axis=0); dsex[N:] = sex_
        
        N += len(spec_)

if before_after_prediction=='before':
    print('\nRemember to run this step again after step3 using argument "after"!!\n')

