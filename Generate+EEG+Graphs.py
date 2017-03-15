# coding: utf-8



import mne
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import stats
from scipy import signal
import os
#os.chdir('C:\\Users\\ning\\Downloads\\eegcode')
#mat10_20 = loadmat('10_20.mat')
#mat10_20 = mat10_20['b']
#mat10_20.shape
from eeg_compute_pipeline import compute_plv_pli_cc,discritized_onset_label_manual,read_annotation

def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    #plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return complex_phase_diff 
def phase_lag_index(data1,data2):
    return np.angle(data1) - np.angle(data2)
"""
threshold_set=[np.arange(0.7,0.9,0.05),#coh
    np.arange(0.05,0.25,0.05),#pli
    np.arange(0.6,0.8,0.05)#plv
    ]
"""
pli_threshold_set = np.arange(0.05,0.3,0.05)
plv_threshold_set = np.arange(0.6,0.8,0.05)
cc_threshold_set = np.arange(0.7,0.9,0.05)
epoch_set = np.arange(2,10,1)


if not os.path.exists('example'):
    os.makedirs('example')
os.chdir('example')
raw_file = 'D:\\NING - spindle\\training set\\suj29_l5nap_day1.fif'
annotation_file = 'D:\\NING - spindle\\training set\\suj29_nap_day1_edited_annotations.txt'

raw = mne.io.read_raw_fif(raw_file,preload=True,)
annotation = pd.read_csv(annotation_file)
montage = "standard_1020"
montage = mne.channels.read_montage(montage)
raw.set_montage(montage)
raw.set_channel_types({'LOc':'eog','ROc':'eog'})
channelList = ['F3','F4','C3','C4','O1','O2']
raw.pick_channels(channelList)
picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
raw.notch_filter(np.arange(60,241,60),picks=picks,filter_length='auto',phase='zero')
raw.filter(1,40,filter_length='auto',phase='zero',h_trans_bandwidth='auto')


for duration in epoch_set:
    
    gold_standard = read_annotation(raw,annotation)
    manual_labels = discritized_onset_label_manual(raw,gold_standard,duration)
    event_array = mne.make_fixed_length_events(raw,id=1,duration=float(duration))
    #event_array[:,1] = duration * raw.info['sfreq']
    #event_array[:,-1] = np.arange(1,len(event_array)+1)
    epochFeatures=compute_plv_pli_cc(raw,duration,plv_threshold_set,pli_threshold_set,cc_threshold_set,manual_labels,channelList)
    if not os.path.exists('feature'):
        os.makedirs('feature')
epochFeatures.to_csv('feature\\feature_%.2f.csv'%(duration))

