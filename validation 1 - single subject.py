# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:40:43 2016

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
from sklearn.metrics import roc_curve
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
    discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=spindle_segment+0.5)
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
    discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=spindle_segment+0.5)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for kk,spindle in enumerate(df['Onset']):
            if spindle_comparison(time_interval,spindle,spindle_duration[kk],spindle_duration_fix=False):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels
            

eegPinelineDesign.change_file_directory('D:\\NING - spindle\\suj13')

had = True
spindle_type = 'fast'
file_to_read = 'suj13_l2nap_day2.fif'
spindle_segment = 2 # define spindle long as 2 seconds for all manual annotations

if had:
    raw = mne.io.read_raw_fif(file_to_read,preload=True,add_eeg_ref=False)
else:
    raw = eegPinelineDesign.load_data('suj13_l2nap_day2.vhdr')
    
"""Setting parameters"""
if spindle_type == 'slow':
    l_freq=10;h_freq=12 # this is how we define a slpw spindle
else:
    l_freq=12.5;h_freq=14.5
channelList = ['F3','F4','C3','C4','O1','O2']
moving_window_size=200;#syn_channels=int(.75 * len(channelList));
l_bound=0.5;h_bound=2; # not using in the example, they are parameters could be reset in the function
thresholds = np.arange(0.1,0.9,0.01);syn_channels = [1,2,3,4,5,6]

manual_spindle = pd.read_csv('suj13_nap_day2_annotations.txt')
manual_spindle = manual_spindle[manual_spindle.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
manual_spindle = manual_spindle[manual_spindle.Onset > 100]
keyword = re.compile('spindle',re.IGNORECASE)
gold_standard = {'Onset':[],'Annotation':[]}
for ii,row in manual_spindle.iterrows():
    if keyword.search(row[-1]):
        gold_standard['Onset'].append(float(row[-2]))
        gold_standard['Annotation'].append(row[-1])
gold_standard = pd.DataFrame(gold_standard)
manual_labels = discritized_onset_label_manual(raw,gold_standard,spindle_segment)
raw.filter(l_freq,h_freq)
raw.pick_channels(channelList)
single_subjuct_result = {}
sensitivity = np.zeros((len(syn_channels),len(thresholds)))
specificity = np.zeros((len(syn_channels),len(thresholds)))
distance = np.zeros((len(syn_channels),len(thresholds)))
fig,ax = plt.subplots(figsize=(20,20))
cnt = 1
for cc,threshold in enumerate(thresholds):
    print('threshold = %.3f' % threshold)
    for bb,syn_channel in enumerate(syn_channels):
        print('  # of channels = %d' % syn_channel)
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
        comparison_matrix = np.vstack((manual_labels,auto_labels)).T
        TN = np.sum((comparison_matrix == [0.,0.]).astype(int),1) == 2
        FP = np.sum((comparison_matrix == [0.,1.]).astype(int),1) == 2
        FN = np.sum((comparison_matrix == [1.,0.]).astype(int),1) == 2
        TP = np.sum((comparison_matrix == [1.,1.]).astype(int),1) == 2
        single_subjuct_result[str(cc)+','+str(bb)]={'correct reject':TN,
                                        'false alarm'   :FP,
                                        'miss'          :FN,
                                        'hit'           :TP}
        sensitivity[bb,cc] = TP.sum()/(TP.sum() + FN.sum())
        specificity[bb,cc] = TN.sum()/(FP.sum() + TN.sum())
        fpr, tpr, _ = roc_curve(manual_labels, auto_labels)
        ax.plot(fpr, tpr)
        distance[bb,cc] = np.sum((manual_labels - auto_labels)**2)
        #plt.close('all')
ax.set(xlabel='false positive rate',ylabel='true positive rate',title='ROC variates over 2 parameters: thresholds and channel numbers')
fig.savefig('ROC variates over 2 parameters_ thresholds and channel numbers.png')
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,sensitivity,shading='gouraud',cmap=plt.cm.Blues)
plt.colorbar(T)
ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='Heat map of sensitivity, thresholds and number of channels')
fig.savefig('Heat map of sensitivity, thresholds and number of channels.png')       
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,1-specificity,shading='gouraud',cmap=plt.cm.Blues)
plt.colorbar(T)
ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='Heat map of false alarm rate (1-specificity), thresholds and number of channels')
fig.savefig('Heat map of false alarm rate (1-specificity), thresholds and number of channels.png')
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,distance,shading='gouraud',cmap=plt.cm.Blues)
plt.colorbar(T)
ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='Heat map of raw difference, thresholds and number of channels')
fig.savefig('Heat map of raw difference, thresholds and number of channels.png')
fig,ax = plt.subplots(1,2,figsize=(20,20))
for ii in range(6):
    mean_sen_over_channels,mean_spe_over_channels = sensitivity[ii,:],specificity[ii,:]
    ax[0].plot([0, 1], [0, 1], 'k--')
    ax[0].plot(1-mean_spe_over_channels,mean_sen_over_channels,label=ii+1)#,xerr=std_spe_over_channels,yerr=std_sen_over_channels)
ax[0].legend()
for ii in range(len(thresholds)):
    mean_sen_over_thresholds,mean_spe_over_thresholds = sensitivity[:,ii],specificity[:,ii]
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].plot(1-mean_spe_over_thresholds,mean_sen_over_thresholds,label=ii)#,xerr=std_spe_over_thresholds,yerr=std_sen_over_thresholds)
ax[1].legend()
std_sen_over_channels,std_spe_over_channels = sensitivity.std(0),specificity.std(0)
std_sen_over_thresholds,std_spe_over_thresholds = sensitivity.std(1),specificity.std(1)






