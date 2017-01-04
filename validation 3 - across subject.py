# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 11:40:01 2017

@author: install
"""

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
            #print(time_interval,spindle,spindle_segment)
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
thresholds = np.arange(0.1,0.95,0.01);syn_channels = [1,2,3,4,5,6]
Ratio_tpr_fpr = np.zeros((len(thresholds),len(syn_channels)))