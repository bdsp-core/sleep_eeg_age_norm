import datetime
import time
import os
import os.path
import re
import subprocess
import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
import mne
#MATLAB_DIRECTORY = '/home/sunhaoqi/matlab'
MATLAB_DIRECTORY = '/usr/local'
MATLAB_BIN_PATH = os.path.join(MATLAB_DIRECTORY,'bin','matlab')


def load_MGH_dataset(data_path, label_path, channels=None):

    ff = sio.loadmat(data_path)
    data_path = os.path.basename(data_path)
    if 's' not in ff:
        raise Exception('No EEG signal found in %s.'%data_path)
    EEG = ff['s']
    EEG = EEG.astype(float)
    channel_names = [ff['hdr'][0,i]['signal_labels'][0].upper().replace('M','A') for i in range(ff['hdr'].shape[1])]
    Fs = 200.

    # load labels
    with h5py.File(label_path, 'r') as ffl:
        sleep_stage = ffl['stage'][()].flatten()
        time_str_elements = ffl['features']['StartTime'][()].flatten()
        start_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
        time_str_elements = ffl['features']['EndTime'][()].flatten()
        end_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
        
    # check signal length = sleep stage length
    assert sleep_stage.shape[0]==EEG.shape[1], 'Inconsistent sleep stage length (%d) and signal length (%d) in %s'%(sleep_stage.shape[0],EEG.shape[1],data_path)

    # check channel number
    assert EEG.shape[0]==len(channel_names), 'Inconsistent channel number in %s'%data_path
    
    # only take EEG channels to study
    if channels is None:
        EEG_channel_ids = list(range(len(channel_names)))
    else:
        EEG_channel_ids = []
        for i in range(len(channels)):
            channel_name_pattern = re.compile(channels[i].upper())#[:2].upper()+'-*'+channels[i][-2:].upper()+'|E[CK]G')
            found = False
            for j in range(len(channel_names)):
                if channel_name_pattern.match(channel_names[j].upper()):
                    EEG_channel_ids.append(j)
                    found = True
                    break
            if not found:
                raise Exception('Channel %s is not found.'%channels[i])
        EEG = EEG[EEG_channel_ids,:]

    # check whether the EEG signal contains NaN
    assert not np.any(np.isnan(EEG)), 'Found Nan in EEG signal in %s'%data_path

    # check whether sleep_stage contains all 5 stages
    stages = np.unique(sleep_stage[np.logical_not(np.isnan(sleep_stage))]).astype(int).tolist()
    assert len(stages)>1, '#sleep stage <= 1: %s in %s'%(stages,data_path)

    params = {'Fs':Fs}
    
    return EEG, sleep_stage, params


def load_CHIMES_dataset(data_path, label_path, channels=[]):

    EEG, params = read_edf(data_path, channels=channels)
    Fs = params['Fs']

    # load labels
    stage_mapping = {'W':5,
                    'I':0,
                    'A':4,
                    'Q':3,}
    df = pd.read_csv(label_path, sep='\t')
    times = df.time.values
    stages = df.stage.values
    
    # generate labels for each sampling point
    start_datetime = params['start_datetime']
    label_times = []
    oneday = datetime.timedelta(days=1)
    for x in times:
        # pad 0 to hour
        xx = x.split(':')
        xx[0] = '%02d'%int(xx[0])  # e.g. 04:53:00 AM
        xx = ':'.join(xx).upper()
        # generate label time by first adding date and then convert to datetime
        # be careful because there is no date in label time:
        # start_datetime is PM --> label_time PM: add start_datetime.date; label_time AM: add start_datetime.date + 1day
        # start_datetime is AM --> label_time AM: add start_datetime.date (won't last to PM)
        if start_datetime.hour>=12:
            if 'PM' in xx:
                xx = start_datetime.strftime('%Y-%m-%d ') + xx
            elif 'AM' in xx:
                xx = (start_datetime+oneday).strftime('%Y-%m-%d ') + xx
            else:
                raise ValueError('No AM/PM in %s'%xx)
        else:
            if 'AM' in xx:
                xx = start_datetime.strftime('%Y-%m-%d ') + xx
            else:
                raise ValueError('PM in %s'%xx)
        label_times.append(datetime.datetime.strptime(xx, '%Y-%m-%d %I:%M:%S %p'))
    assert np.all(np.array(list(map(lambda x:x.total_seconds(),np.diff(label_times))))>0)
    
    sleep_stage = np.zeros(EEG.shape[1])+np.nan
    window_size = int(round(30*Fs))
    for i, x in enumerate(label_times):
        startid = int(round((x-start_datetime).total_seconds()*Fs))
        endid = startid+window_size
        if startid<0:
            continue#raise ValueError('startid < 0')
        if startid>=EEG.shape[1]:
            continue#raise ValueError('startid >= length')
        if endid<0:
            continue#raise ValueError('endid < 0')
        if endid>=EEG.shape[1]:
            continue#raise ValueError('endid >= length')
            
        sleep_stage[startid:endid] = stage_mapping[stages[i]]
            
    return EEG, sleep_stage, params


def load_ChicagoPediatric_dataset(data_path, label_path, channels=[]):

    EEG, params = read_edf(data_path, channels=channels)
    Fs = params['Fs']
    signal *= 1e6
    
    if label_path=='none':
        sleep_stage = None
    else:
        # load labels
        stage_mapping = {'SLEEP-S0':5,
                        'SLEEP-REM':4,
                        'SLEEP-S1':3,
                        'SLEEP-S2':2,
                        'SLEEP-S3':1,
                        'SLEEP-UNSCORED':np.nan}
        df = pd.read_csv(label_path, sep='\t', skiprows=17)
        times = df['Time [hh:mm:ss]'].values
        stages = df['Sleep Stage'].values
        assert np.all(stages==df['Event']), 'Sleep Stage != Event in label file %s'%label_path
        
        # generate labels for each sampling point
        start_datetime = params['start_datetime']
        label_times = []
        oneday = datetime.timedelta(days=1)
        for x in times:
            # pad 0 to hour
            xx = x.split(':')
            xx[0] = '%02d'%int(xx[0])  # e.g. 04:53:00 AM
            xx = ':'.join(xx).upper()
            # generate label time by first adding date and then convert to datetime
            # be careful because there is no date in label time:
            # start_datetime is PM --> label_time PM: add start_datetime.date; label_time AM: add start_datetime.date + 1day
            # start_datetime is AM --> label_time AM: add start_datetime.date (won't last to PM)
            if start_datetime.hour>=12:
                if 'PM' in xx:
                    xx = start_datetime.strftime('%Y-%m-%d ') + xx
                elif 'AM' in xx:
                    xx = (start_datetime+oneday).strftime('%Y-%m-%d ') + xx
                else:
                    raise ValueError('No AM/PM in %s'%xx)
            else:
                if 'AM' in xx:
                    xx = start_datetime.strftime('%Y-%m-%d ') + xx
                else:
                    raise ValueError('PM in %s'%xx)
            label_times.append(datetime.datetime.strptime(xx, '%Y-%m-%d %I:%M:%S %p'))
        assert np.all(np.array(list(map(lambda x:x.total_seconds(),np.diff(label_times))))>0)
        
        sleep_stage = np.zeros(EEG.shape[1])+np.nan
        window_size = int(round(30*Fs))
        for i, x in enumerate(label_times):
            startid = int(round((x-start_datetime).total_seconds()*Fs))
            endid = startid+window_size
            if startid<0:
                continue#raise ValueError('startid < 0')
            if startid>=EEG.shape[1]:
                continue#raise ValueError('startid >= length')
            if endid<0:
                continue#raise ValueError('endid < 0')
            if endid>=EEG.shape[1]:
                continue#raise ValueError('endid >= length')
                
            sleep_stage[startid:endid] = stage_mapping[stages[i]]
    
    return EEG, sleep_stage, params
    
 
def read_edf(data_path, channels=[]):
    edf           = mne.io.read_raw_edf(data_path, verbose=False, stim_channel=None)
    channel_names = edf.info['ch_names']
    #channel_names = [x.replace('M','A') for x in channel_names]#TODO make sure use re in channels
    Fs            = edf.info['sfreq']
    start_datetime= edf.info['meas_date'].replace(tzinfo=None)#datetime.datetime(*time.gmtime(edf.info['meas_date'][0])[:6])
    #signal  = edf.to_data_frame()
    #signal  = signal.as_matrix()

    if len(channels)>0:
        #EEG_channel_ids = list(range(len(channel_names)))
        #else:
        EEG_channel_ids = []
        for i in range(len(channels)):
            channel_name_pattern = re.compile(channels[i].upper())#[:2].upper()+'-*'+channels[i][-2:].upper()+'|E[CK]G')
            found = False
 
            for j in range(len(channel_names)):
                if channel_name_pattern.match(channel_names[j].upper()):
                    EEG_channel_ids.append(j)
                    found = True
                    break
            assert found, 'Channel %s is not be found'%channels[i]
        signal = edf.get_data(picks=np.array(channel_names)[EEG_channel_ids])
    else:
        signal = edf.get_data()
     
    """
    matlab_input = 'edf_reader_matlab_input.mat'
    matlab_output = 'edf_reader_matlab_output.mat'
    matlab_log = 'matlab_log.txt'
    if os.path.exists(matlab_output):
        os.remove(matlab_output)
        
    sio.savemat(matlab_input, {'data_path':data_path, 'channels':channels})
    with open(matlab_log,'w') as ff:
        subprocess.check_call([MATLAB_BIN_PATH, '-nodisplay', '-nodesktop', '-r', 'edf_reader_matlab.m'], stdout=ff, stderr=ff)
    assert os.path.exists(matlab_output), 'data_path is bad. Refer to %s.'%matlab_log
    res = sio.loadmat(matlab_output)
    os.remove(matlab_output)
    os.remove(matlab_input)
    os.remove(matlab_log)
    
    start_datetime= datetime.datetime.strptime(res['start_datetime'][0], '%y.%m.%d-%H.%M.%S')
    signal  = res['record']
    channel_names = [str(res['channel_names'][0,ii][0]) for ii in range(res['channel_names'].shape[1])]
    Fs            = res['Fs'].flatten()
    if len(Fs)==1:
        Fs = Fs[0]
    """

    params  = {'Fs': Fs, 'start_datetime':start_datetime}
 
    return signal.astype(float),  params
    

if __name__ == "__main__":
    import pdb
    pdb.set_trace()
    EEG, params = load_CHIMES_dataset('/data/Dropbox (Partners HealthCare)/MGH_Neonatal_EEG/CHIMES/10005.EDF', '/data/sleep_age_descriptive/mycode/data/ARSL_ST.csv', channels=['C4A1','C3A2','O2A1','O1A2'])
    print(EEG.shape)
