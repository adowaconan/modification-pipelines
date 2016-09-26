# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:52:52 2016

@author: install
"""

import eegPinelineDesign
import pandas as pd
import numpy as np
import mne
import re
import pickle
import matplotlib.pyplot as plt
chan_dict={'Ch56': 'TP8', 'Ch61': 'F6', 'Ch3': 'F3', 'Ch45': 'P1', 'Ch14': 'P3', 
           'Ch41': 'C1', 'Ch1': 'Fp1', 'Ch46': 'P5', 'Ch7': 'FC1', 'Ch37': 'F5', 
           'Ch21': 'TP10', 'Ch8': 'C3', 'Ch11': 'CP5', 'Ch28': 'FC6', 'Ch17': 'Oz', 
           'Ch39': 'FC3', 'Ch38': 'FT7', 'Ch58': 'C2', 'Ch33': 'AF7', 'Ch48': 'PO3', 
           'Ch9': 'T7', 'Ch49': 'POz', 'Ch2': 'Fz', 'Ch15': 'P7', 'Ch20': 'P8', 
           'Ch60': 'FT8', 'Ch57': 'C6', 'Ch32': 'Fp2', 'Ch29': 'FC2', 'Ch59': 'FC4', 
           'Ch35': 'AFz', 'Ch44': 'CP3', 'Ch47': 'PO7', 'Ch30': 'F4', 'Ch62': 'F2', 
           'Ch4': 'F7', 'Ch24': 'Cz', 'Ch31': 'F8', 'Ch64': 'ROc', 'Ch23': 'CP2', 
           'Ch25': 'C4', 'Ch40': 'FCz', 'Ch53': 'P2', 'Ch19': 'P4', 'Ch27': 'FT10', 
           'Ch50': 'PO4', 'Ch18': 'O2', 'Ch55': 'CP4', 'Ch6': 'FC5', 'Ch12': 'CP1', 
           'Ch16': 'O1', 'Ch52': 'P6', 'Ch5': 'FT9', 'Ch42': 'C5', 'Ch36': 'F1', 
           'Ch26': 'T8', 'Ch51': 'PO8', 'Ch34': 'AF3', 'Ch22': 'CP6', 'Ch54': 'CPz', 
           'Ch13': 'Pz', 'Ch63': 'LOc', 'Ch43': 'TP7'}
def rescale(M):
    M = np.array(M)
    return (M - M.min())/(M.max() - M.min())+2
had = False
current_working_folder=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set\\')
"""
raw = mne.io.read_raw_brainvision('suj5_d2_nap.vhdr',scale=1e6,preload=True)

raw.rename_channels(chan_dict)
chan_list=raw.ch_names[:2]
if 'LOc' not in chan_list:
    chan_list.append('LOc')
if 'ROc' not in chan_list:
    chan_list.append('ROc')

raw.pick_channels(chan_list)
raw.filter(1,200)
picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
#noise_cov=mne.compute_raw_covariance(raw.add_eeg_average_proj(),picks=picks)
raw.notch_filter(np.arange(60,241,60), picks=picks)
#reject = dict(eeg=eegReject)

ica = mne.preprocessing.ICA(n_components=20,max_iter=3000,noise_cov=None, random_state=0)
ica.fit(raw,picks=picks,decim=2,tstep=2.)
ica.detect_artifacts(raw,eog_ch=['LOc', 'ROc'],eog_criterion=0.4,
                     skew_criterion=2,kurt_criterion=2,var_criterion=2)
"""
subjectList=[5,6,8,9,10]
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files) and ('suj%d'%idx in files)]
    print(list_file_to_read);low=1;high=200
    for file_to_read in list_file_to_read:
        if had:
            raw = mne.io.read_raw_fif('D:\\NING - spindle\\training set\\'+file_to_read[:-5] + '.fif',preload=True,add_eeg_ref=False)
        else:
            try:
                raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=180,eogReject=300,n_ch=-2)
            except:
                try:
                    raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=240,eogReject=300,n_ch=-2)
                except:
                    try:
                        raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=300,eogReject=300,n_ch=-2)
                    except:
                        raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=360,eogReject=300,n_ch=-2)
            raw.save(file_to_read[:-5] + '.fif',overwrite=True)
