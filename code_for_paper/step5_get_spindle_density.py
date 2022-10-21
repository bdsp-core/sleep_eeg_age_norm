"""
luna-base version v0.25.5 (release date 24-May-2021)
luna-base build date/time Jul 22 2021 00:36:18

> library(luna)
** lunaR v0.25.2 23-Feb-2021
** luna v0.25.5 31-Mar-2021
"""
import os
import pickle
import subprocess
import shutil
import numpy as np
import scipy.io as sio
import h5py
import pandas as pd
import pyedflib
from tqdm import tqdm
import mne
import sys
sys.path.insert(0, '/data/brain_age_descriptive/mycode')
from read_dataset import *
from step5_get_spindle_density import convert_edf, detect_spindle

    
    
if __name__=='__main__':
    channel = 'C'
    assert channel in ['C', 'F']
    
    # get age vs peak freq
    # to have a rough central frequency for detection
    if channel=='C':
        path = 'step4_results_row.pickle'
    else:
        path = 'step4_results_row_MGH_frontal.pickle'
    with open(path, 'rb') as ff:
        res = pickle.load(ff)
    xages = res[3]
    freq = res[2]
    id10 = np.argmin(np.abs(freq-10))
    id16 = np.argmin(np.abs(freq-16))
    
    pt1 = 28    # spindle starts at 6 weeks
    pt2 = 158   # neonatal data ends at 8 month
    pt4 = 960   # fast increase starts
    pt5 = np.where(xages>=20)[0][0]  # stable afterwards
        
    sexs = ['male', 'female']
    sex_txt2num = {'male':1, 'female':0}
    peak_freqs_dict = {}
    peak_freqs_raw_dict = {}
    for sex in sexs:
        if sex=='female':
            pt3 = 434   # slow increase starts
        else:
            pt3 = 351   # slow increase starts
        
        spec = res[0][(channel,'N2',sex_txt2num[sex])]
        dspec = spec[:,id10:id16].T - ((spec[:,id16] - spec[:,id10])/(freq[id16] - freq[id10])*(freq[id10:id16].reshape(-1,1) - freq[id10])+spec[:,id10])
        
        #from scipy.interpolate import UnivariateSpline
        peak_freqs_raw = freq[id10+np.argmax(dspec,axis=0)]
        #foo = UnivariateSpline(xages[pt1:], peak_freqs[pt1:], w=None, k=3, s=75)
        #peak_freqs = foo(xages[pt1:])
        peak_freqs = np.r_[
            np.zeros(pt1)+np.nan,
            np.zeros(pt2-pt1)+np.median(peak_freqs_raw[pt1:pt2]),
            (np.arange(pt2, pt3)-pt2)*(peak_freqs_raw[pt3]-peak_freqs_raw[pt2])/(pt3-pt2)+peak_freqs_raw[pt2],
            (np.arange(pt3, pt4)-pt3)*(peak_freqs_raw[pt4]-peak_freqs_raw[pt3])/(pt4-pt3)+peak_freqs_raw[pt3],
            (np.arange(pt4, pt5)-pt4)*(peak_freqs_raw[pt5]-peak_freqs_raw[pt4])/(pt5-pt4)+peak_freqs_raw[pt4],
            np.zeros(len(xages)-pt5)+np.median(peak_freqs_raw[pt5:]),
            ]
        peak_freqs_dict[sex] = peak_freqs
        peak_freqs_raw_dict[sex] = peak_freqs_raw
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        plt.close()
        fig = plt.figure(figsize=(13,10))
        gs = GridSpec(3,1,height_ratios=[1,3,3])
        ax = fig.add_subplot(gs[0,0]); ax0=ax
        ax.plot(np.arange(len(xages)),xages, c='k', lw=2)
        ax.set_xlabel('Age ID')
        ax.set_ylabel('Age (year)')
        ax = fig.add_subplot(gs[1,0], sharex=ax0)
        ax.imshow(spec.T,aspect='auto',origin='lower',extent=(0,len(xages),freq.min(),freq.max()), cmap='turbo', vmin=-5, vmax=25)
        ax.plot(np.arange(len(xages)),peak_freqs, c='r', lw=2)
        ax.plot(np.arange(len(xages)),peak_freqs_raw, c='k', lw=2)
        ax.axvline(pt1, c='k', ls='--')
        ax.axvline(pt2, c='k', ls='--')
        ax.axvline(pt3, c='k', ls='--')
        ax.axvline(pt4, c='k', ls='--')
        ax.axvline(pt5, c='k', ls='--')
        ax.set_xlabel('Age ID')
        ax.set_ylabel('frequency (Hz)')
        ax = fig.add_subplot(gs[2,0], sharex=ax0)
        ax.imshow(dspec, aspect='auto', origin='lower', extent=(0,len(xages),freq[id10],freq[id16]), cmap='turbo', vmin=-0.5, vmax=4)
        ax.plot(np.arange(len(xages)),peak_freqs, c='r', lw=2)
        ax.plot(np.arange(len(xages)),peak_freqs_raw, c='k', lw=2)
        ax.axvline(pt1, c='k', ls='--')
        ax.axvline(pt2, c='k', ls='--')
        ax.axvline(pt3, c='k', ls='--')
        ax.axvline(pt4, c='k', ls='--')
        ax.axvline(pt5, c='k', ls='--')
        ax.set_xlabel('Age ID')
        ax.set_ylabel('frequency (Hz)')
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'spindle_detection_cfreq_initial_value_{channel}_{sex}.png')
        """
    peak_freqs = (peak_freqs_dict['male']+peak_freqs_dict['female'])/2
    """
    plt.close()
    fig = plt.figure(figsize=(13,7))
    ax = fig.add_subplot(111); ax0=ax
    ax.plot(xages,peak_freqs_dict['male'], c='b', lw=2, alpha=0.4)
    ax.plot(xages,peak_freqs_raw_dict['male'], c='b', lw=2, ls=':', alpha=0.4)
    #ax.set_xlabel('Age (year)')
    #ax.set_ylabel('frequency (Hz)')
    #ax = fig.add_subplot(312, sharex=ax0)
    ax.plot(xages,peak_freqs_dict['female'], c='r', lw=2, alpha=0.4)
    ax.plot(xages,peak_freqs_raw_dict['female'], c='r', lw=2, ls=':', alpha=0.4)
    #ax.set_xlabel('Age (year)')
    #ax.set_ylabel('frequency (Hz)')
    #ax = fig.add_subplot(313, sharex=ax0)
    ax.plot(xages,peak_freqs, c='k', lw=2)
    ax.plot(xages,peak_freqs, c='k', lw=2)
    ax.set_xlabel('Age (year)')
    ax.set_ylabel('frequency (Hz)')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'spindle_detection_cfreq_initial_value_{channel}_avg.png')
    """
        
    xages = xages[::10]
    peak_freqs = peak_freqs[::10]
    pt5 = np.where(xages>=20)[0][0]
    xages = np.r_[xages[:pt5+1], xages[-1]]
    peak_freqs = np.r_[peak_freqs[:pt5+1], peak_freqs[-1]]
    
    
    # get all file paths
    output_dir = '/data/brain_age_descriptive/mycode-NBA-revision/step5_alpha_and_spindle_peak_freq'
    os.makedirs(output_dir, exist_ok=True)
    
    edf_xml_path = '/data/brain_age_descriptive/edf_xml_for_luna'
    os.makedirs(edf_xml_path, exist_ok=True)
    
    MGH_data_list = pd.read_csv('/data/brain_age_descriptive/mycode/data/MGH_data_list.txt', sep='\t')
    MGH_data_list = MGH_data_list[MGH_data_list.state=='good'].reset_index(drop=True)
    MGH_err_subjects = pd.read_csv('/data/brain_age_descriptive/mycode/data/MGH_err_subject_reason.txt', sep='::: ', header=None)[0].values
    
    ChicagoPediatric_data_list = pd.read_csv('/data/brain_age_descriptive/mycode/data/ChicagoPediatric_data_list.txt', sep='\t')
    ChicagoPediatric_data_list = ChicagoPediatric_data_list[ChicagoPediatric_data_list.state=='good'].reset_index(drop=True)
    ChicagoPediatric_err_subjects = pd.read_csv('/data/brain_age_descriptive/mycode/data/ChicagoPediatric_err_subject_reason.txt', sep='::: ', header=None)[0].values

    CHIMES_data_list = pd.read_csv('/data/brain_age_descriptive/mycode/data/CHIMES_data_list.txt', sep='\t')
    CHIMES_data_list = CHIMES_data_list[CHIMES_data_list.state=='good'].reset_index(drop=True)
    CHIMES_err_subjects = pd.read_csv('/data/brain_age_descriptive/mycode/data/CHIMES_err_subject_reason.txt', sep='::: ', header=None)[0].values
    # add age to CHIMES
    #!! need to modify edf.py to get the correct meas_date
    ages_weeks = np.array([(mne.io.read_raw_edf(CHIMES_data_list.signal_file.iloc[i], preload=False, verbose=False).info['meas_date'].replace(tzinfo=None) - datetime.datetime.strptime(CHIMES_data_list.dob.iloc[i], '%Y-%m-%d %H:%M:%S')).total_seconds()/24/3600/7 for i in range(len(CHIMES_data_list))])
    # remove CHIMES age < 6 weeks
    ids = ages_weeks>=6
    CHIMES_data_list = CHIMES_data_list[ids].reset_index(drop=True)
    ages_chimes = ages_weeks[ids]*7/365
    
    if channel=='C':
        MGH_channels = np.array(['C3A2','C4A1'])
        ChicagoPediatric_channels = np.array(['C3-M2','C4-M1'])
        CHIMES_channels = np.array(['C3A2','C4A1'])
    elif channel=='F':
        MGH_channels = np.array(['F3A2','F4A1'])
    else:
        raise NotImplementedError(channel)

    """
    ###########################################################################
    edf_paths = []
    xml_paths = []
    ages = []
    for ii in tqdm(range(len(MGH_data_list))):
        if os.path.basename(MGH_data_list.feature_file[ii]) in MGH_err_subjects:
            continue
            
        #try:
        Fs = 200
        basename = os.path.basename(MGH_data_list.signal_file[ii]).replace('.mat', '')
        
        edf_path = os.path.join(edf_xml_path, basename.replace(',','')+'.edf')
        xml_path = os.path.join(edf_xml_path, basename.replace(',','')+'.xml')
        if os.path.exists(edf_path) and os.path.exists(xml_path):
            edf_paths.append(edf_path)
            xml_paths.append(xml_path)
            ages.append(MGH_data_list.age.iloc[ii])
            continue
            
        # create edf
        convert_edf(MGH_data_list.signal_file[ii], edf_path, Fs=Fs, channels=MGH_channels)
        
        # create sleep stage xml
        dst_dir = '/home/sunhaoqi'
        subprocess.check_call(['rclone', 'copy', MGH_data_list.label_file[ii].replace('/media/mad3/Projects_New/', 'dropbox:'), dst_dir])
        tmp = os.path.join(dst_dir, os.path.basename(MGH_data_list.label_file[ii]))
        with h5py.File(tmp, 'r') as ff:
            sleep_stages = ff['stage'][:].flatten()
        os.remove(tmp)
        sleep_stages = sleep_stages[np.arange(0, len(sleep_stages), Fs*30)]
        sleep_stages[np.isnan(sleep_stages)] = -1
        sleep_stage_mapping = {-1:0, 0:0, 5:0, 4:5, 3:1, 2:2, 1:3}
        with open(xml_path, 'w') as ff:
            ff.write('<CMPStudyConfig>\n')
            ff.write('<EpochLength>30</EpochLength>\n')
            ff.write('<SleepStages>\n')
            for ss in sleep_stages:
                ff.write('<SleepStage>%d</SleepStage>\n'%sleep_stage_mapping[ss])
            ff.write('</SleepStages>\n')
            ff.write('</CMPStudyConfig>')
            
        #except Exception as ee:
        #    print('%s: %s'%(basename, ee.message))
        #    continue
        
        edf_paths.append(edf_path)
        xml_paths.append(xml_path)
        ages.append(MGH_data_list.age.iloc[ii])
    
    ages = np.array(ages)
    edf_paths = np.array(edf_paths)
    xml_paths = np.array(xml_paths)
    
    for ai in range(len(xages)-1):
        save_path = os.path.join(output_dir, 'luna_output_MGH_%s_%g-%gyr.xlsx'%(channel, xages[ai],xages[ai+1]))
        if os.path.exists(save_path):
            continue
        min_age = xages[ai]
        max_age = xages[ai+1]
        if ii==len(xages)-2:
            max_age = max_age+0.001
        age_group_ids = np.where((ages>=min_age) & (ages<max_age))[0]
        if len(age_group_ids)==0:
            continue
        
        cfreq = peak_freqs[ai]
        if cfreq is None or np.isnan(cfreq):
            continue
        df = detect_spindle(edf_xml_path, edf_paths[age_group_ids], xml_paths[age_group_ids],
                            MGH_channels,
                            cfreq=cfreq)
        df.to_excel(save_path, index=False)
    if channel!='C':
        raise SystemExit
    
    ###########################################################################
    edf_paths = []
    xml_paths = []
    ages = []
    for ii in tqdm(range(len(ChicagoPediatric_data_list))):
        if os.path.basename(ChicagoPediatric_data_list.feature_file[ii]) in ChicagoPediatric_err_subjects:
            continue
        #try:
        basename = os.path.basename(ChicagoPediatric_data_list.signal_file[ii]).replace('.edf', '').replace(' ','')
        
        edf_path = os.path.join(edf_xml_path, basename+'.edf')
        xml_path = os.path.join(edf_xml_path, basename+'.xml')
        if os.path.exists(edf_path) and os.path.exists(xml_path):
            edf_paths.append(edf_path)
            xml_paths.append(xml_path)
            ages.append(ChicagoPediatric_data_list.age.iloc[ii])
            continue
        
        # create sleep stage xml
        if os.path.exists(ChicagoPediatric_data_list.feature_file[ii]):
            res = sio.loadmat(ChicagoPediatric_data_list.feature_file[ii], variable_names=['sleep_stages', 'predicted_sleep_stages_smoothed', 'seg_times'])
            if 'predicted_sleep_stages_smoothed' in res:
                sleep_stages = np.argmax(res['predicted_sleep_stages_smoothed'], axis=1)+1
            elif 'sleep_stages' in res:
                sleep_stages = res['sleep_stages'].flatten()
            else:
                continue
        else:
            if ChicagoPediatric_data_list.label_file[ii]=='none':
                continue
            _, sleep_stages, params = load_ChicagoPediatric_dataset(ChicagoPediatric_data_list.signal_file[ii], ChicagoPediatric_data_list.label_file[ii], channels=ChicagoPediatric_channels)
            Fs = int(round(params['Fs']))
            sleep_stages = sleep_stages[np.arange(0, len(sleep_stages), Fs*30)]
            
        sleep_stages[np.isnan(sleep_stages)] = -1
        sleep_stage_mapping = {-1:0, 0:0, 5:0, 4:5, 3:1, 2:2, 1:3}
        with open(xml_path, 'w') as ff:
            ff.write('<CMPStudyConfig>\n')
            ff.write('<EpochLength>30</EpochLength>\n')
            ff.write('<SleepStages>\n')
            for ss in sleep_stages:
                ff.write('<SleepStage>%d</SleepStage>\n'%sleep_stage_mapping[ss])
            ff.write('</SleepStages>\n')
            ff.write('</CMPStudyConfig>')
            
        # get edf
        #try:
        #    os.unlink(edf_path)
        #except Exception as ee:
        #    pass
        os.symlink(ChicagoPediatric_data_list.signal_file[ii], edf_path)
            
        edf_paths.append(edf_path)
        xml_paths.append(xml_path)
        ages.append(ChicagoPediatric_data_list.age.iloc[ii])
            
        #except Exception as ee:
        #    print('%s: %s'%(basename, ee.message))
        #    continue
        
    ages = np.array(ages)
    edf_paths = np.array(edf_paths)
    xml_paths = np.array(xml_paths)
    
    for ai in range(len(xages)-1):
        min_age = xages[ai]
        max_age = xages[ai+1]
        if ii==len(xages)-2:
            max_age = max_age+0.001
        age_group_ids = np.where((ages>=min_age) & (ages<max_age))[0]
        if len(age_group_ids)==0:
            continue
        
        cfreq = peak_freqs[ai]
        if cfreq is None or np.isnan(cfreq):
            continue
        df = detect_spindle(edf_xml_path, edf_paths[age_group_ids], xml_paths[age_group_ids],
                            ChicagoPediatric_channels,
                            cfreq=cfreq)
        df.to_excel(os.path.join(output_dir, 'luna_output_ChicagoPediatric_%s_%g-%gyr.xlsx'%(channel, xages[ai],xages[ai+1])), index=False)
    """
    
    
    ###########################################################################
    edf_paths = []
    xml_paths = []
    for ii in tqdm(range(len(CHIMES_data_list))):
        if os.path.basename(CHIMES_data_list.feature_file[ii]) in CHIMES_err_subjects:
            continue
        #try:
        basename = os.path.basename(CHIMES_data_list.signal_file[ii]).replace('.EDF', '')
        
        edf_path = os.path.join(edf_xml_path, basename+'.edf')
        xml_path = os.path.join(edf_xml_path, basename+'.xml')
        if os.path.exists(edf_path) and os.path.exists(xml_path):
            edf_paths.append(edf_path)
            xml_paths.append(xml_path)
            continue
                
        # create edf
        convert_edf(CHIMES_data_list.signal_file[ii], edf_path, channels=CHIMES_channels)
                
        # create sleep stage xml
        _, sleep_stages, params = load_CHIMES_dataset(CHIMES_data_list.signal_file[ii], CHIMES_data_list.label_file[ii], channels=CHIMES_channels)
        Fs = int(round(params['Fs']))
        sleep_stages = sleep_stages[np.arange(0, len(sleep_stages), Fs*30)]
        sleep_stages[np.isnan(sleep_stages)] = -1
        sleep_stage_mapping = {-1:0, 0:0, 5:0, 4:5, 3:2}
        with open(xml_path, 'w') as ff:
            ff.write('<CMPStudyConfig>\n')
            ff.write('<EpochLength>30</EpochLength>\n')
            ff.write('<SleepStages>\n')
            for ss in sleep_stages:
                ff.write('<SleepStage>%d</SleepStage>\n'%sleep_stage_mapping[ss])
            ff.write('</SleepStages>\n')
            ff.write('</CMPStudyConfig>')
            
        edf_paths.append(edf_path)
        xml_paths.append(xml_path)
            
        #except Exception as ee:
        #    print('%s: %s'%(basename, ee.message))
        #    continue
    cfreq = np.nanmedian(peak_freqs[xages<=ages_chimes.max()])
    df = detect_spindle(edf_xml_path, edf_paths, xml_paths, CHIMES_channels, cfreq=cfreq)
    df.to_excel(os.path.join(output_dir, 'luna_output_CHIMES_%s.xlsx'%channel), index=False)
    
