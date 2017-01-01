# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:06:23 2016

@author: install
"""
import os
os.chdir('C:\\Users\\ning\\OneDrive\\python works\\modification-pipelines')
import eegPinelineDesign
from eegPinelineDesign import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_curve, auc
import re
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    if spindle_duration_fix:
        spindle_start = spindle
        spindle_end   = spindle + spindle_duration
        a =  np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
    else:
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        a = np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
def discritized_onset_label_manual(raw,df,spindle_segment):
    discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            print(time_interval,spindle,spindle_segment)
            if spindle_comparison(time_interval,spindle,spindle_segment):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels
def discritized_onset_label_auto(raw,df,spindle_segment):
    spindle_duration = df['Duration'].values
    discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for kk,spindle in enumerate(df['Onset']):
            if spindle_comparison(time_interval,spindle,spindle_duration[kk],spindle_duration_fix=False):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels

def read_annotation(file_to_read,raw,file_in_fold):
    annotation_file = [files for files in file_in_fold if('txt' in files) and (file_to_read.split('_')[0] in files) and (file_to_read.split('_')[1] in files)]
    manual_spindle = pd.read_csv(annotation_file[0])
    manual_spindle = manual_spindle[manual_spindle.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
    manual_spindle = manual_spindle[manual_spindle.Onset > 100] 
    keyword = re.compile('spindle',re.IGNORECASE)
    gold_standard = {'Onset':[],'Annotation':[]}
    for ii,row in manual_spindle.iterrows():
        if keyword.search(row[-1]):
            gold_standard['Onset'].append(float(row.Onset))
            gold_standard['Annotation'].append(row.Annotation)
    gold_standard = pd.DataFrame(gold_standard) 
    return gold_standard            
try:
    file_in_fold = eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\training set')
except:
    file_in_fold = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]    
"""Setting parameters"""
had = True
spindle_type = 'slow'
spindle_segment = 2 # define spindle long as 2 seconds for all manual annotations

if spindle_type == 'slow':
    l_freq=10;h_freq=12 # this is how we define a slpw spindle
else:
    l_freq=12.5;h_freq=14.5
channelList = ['F3','F4','C3','C4','O1','O2']
moving_window_size=200;#syn_channels=int(.75 * len(channelList));
l_bound=0.5;h_bound=2; # not using in the example, they are parameters could be reset in the function
thresholds = np.arange(0.1,0.95,0.1);syn_channels = [1,2,3,4,5,6]
Ratio_tpr_fpr = np.zeros((len(thresholds),len(syn_channels)))

for aa,threshold in enumerate(thresholds):
    for bb,syn_channel in enumerate(syn_channels):
        manual_label_total = [];auto_label_total=[]
        for file_to_read in list_file_to_read:
            raw = mne.io.read_raw_fif(file_to_read,preload=True)
            raw.pick_channels(channelList)
            raw.filter(l_freq,h_freq,l_trans_bandwidth=0.5,h_trans_bandwidth='auto')
            picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                       stim=False)
            raw.notch_filter(np.arange(60, 241, 60), picks=picks, filter_length='auto',
                 phase='zero')
            gold_standard = read_annotation(file_to_read,raw,file_in_fold)
            
            time_find,mean_peak_power,Duration,peak_time,peak_at = spindle_validation_step1(raw,channelList,file_to_read,
                                                                                         moving_window_size=moving_window_size,
                                                                                         threshold=threshold,
                                                                                         syn_channels=syn_channel,
                                                                                         l_freq=l_freq,
                                                                                         h_freq=h_freq)
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            result.reset_index()
            
            auto_labels = discritized_onset_label_auto(raw,result,spindle_segment)
            auto_label_total = np.concatenate((auto_label_total, auto_labels))
            manual_labels = discritized_onset_label_manual(raw,gold_standard,spindle_segment)
            manual_label_total = np.concatenate((manual_label_total,manual_labels))
            
        fpr, tpr, _ = roc_curve(manual_label_total, auto_label_total)
        Ratio_tpr_fpr[aa,bb] = tpr / fpr
        
variables_to_save = {'ratio':Ratio_tpr_fpr,'thresholds':thresholds,'syn_channels':syn_channels}