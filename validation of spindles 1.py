# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:16:08 2016

@author: install
"""

import eegPinelineDesign
from eegPinelineDesign import *
import numpy as np
import mne
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

"""Setting parameters"""
l_freq=10;h_freq=12 # this is how we define a slpw spindle
channelList = ['F3','F4','C3','C4','O1','O2']
moving_window_size=200;#syn_channels=int(.75 * len(channelList));
l_bound=0.5;h_bound=2; # not using in the example, they are parameters could be reset in the function
thresholds = np.arange(0.1,1.5,0.05);syn_channels = [1,2,3,4,5,6]
cnt = 1
spindle_counts_matrix = []
folderList = change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33))) # manually change the range, the second number is the position after the last stop
for idx in subjectList: 
    
    folder_ = [folder_to_look for folder_to_look in folderList if str(idx) in folder_to_look]
    current_working_folder = change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    #print(list_file_to_read);
    for file_to_read in list_file_to_read:
        
        raw = mne.io.read_raw_fif(file_to_read[:-5] + '.fif',preload=True,add_eeg_ref=False)
        
        raw.filter(l_freq,h_freq)
        raw.pick_channels(channelList)
        single_subjuct_count = np.zeros((len(thresholds),len(syn_channels)))
        print('subject',cnt)
        for cc,threshold in enumerate(thresholds):
            print('threshold = %.3f' % threshold)
            for bb,syn_channel in enumerate(syn_channels):
                print('# of channels = %d' % syn_channel)
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
                
                
                fileName = file_to_read[:-5] + '_slow_spindle.csv'
                single_subjuct_count[cc,bb]=len(result)
                plt.close('all')
        cnt += 1
        spindle_counts_matrix.append(single_subjuct_count)
BIG = np.array(spindle_counts_matrix)
"""Plotting section"""
plt.close('all')
############ plot mean over subjects############
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,BIG.mean(0).T,shading='gouraud')
plt.colorbar(T)
ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='Heat map of spindles detected variating two scales, thresholds and number of channels')
fig.savefig('Heat map mean over subjects.png')     
############ plot std over subjects ##############
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,BIG.std(0).T,shading='gouraud')
plt.colorbar(T)
ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='Heat map of std of spindles detected variating two scales, thresholds and number of channels')
fig.savefig('Heat map std over subjects.png')