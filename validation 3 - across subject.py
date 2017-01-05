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
       
try:
    file_in_fold = eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\allData')
except:
    file_in_fold = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData')
####################################################################################################
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]    
"""Setting parameters"""
had = True
spindle_types = ['slow','fast']
spindle_segment = 2 # define spindle long as 2 seconds for all manual annotations


channelList = ['F3','F4','C3','C4','O1','O2']
moving_window_size=200;#syn_channels=int(.75 * len(channelList));
l_bound=0.5;h_bound=2; # not using in the example, they are parameters could be reset in the function
thresholds = np.arange(0.1,1.005,0.005);syn_channels = [1,2,3,4,5,6]
Ratio_tpr_fpr = np.zeros((len(thresholds),len(syn_channels)))
Predictions = {}
#######################################################################################################
for spindle_type in spindle_types:
    if spindle_type == 'slow':
        l_freq=10;h_freq=12 # this is how we define a slpw spindle
    else:
        l_freq=12.5;h_freq=14.5
    for ii,sample in enumerate(list_file_to_read):
        sub = sample.split('_')[0]
        day = sample.split('_')[-1][:4]
        
        raw = mne.io.read_raw_fif(sample,preload=True,add_eeg_ref=False)
            
        raw.filter(l_freq,h_freq)
        raw.pick_channels(channelList)
        single_subjuct_count = np.zeros((len(thresholds),len(syn_channels)))
        print('subject',sub,'day',day)
        channel_threshold_temp = []
        for bb,syn_channel in enumerate(syn_channels):
            print('# of channels = %d' % syn_channel)
            threshold_temp = []
            for cc,threshold in enumerate(thresholds):
                print('threshold = %.3f' % threshold)
                time_find,mean_peak_power,Duration,peak_time,peak_at = spindle_validation_step1(raw,channelList,sample,
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
                threshold_temp.append(result)
                ###################################################################
                """leave a place for validation"""
                ###################################################################
                
            channel_threshold_temp.append(threshold_temp)
        Predictions[str(sub)+'_'+str(day)]=channel_threshold_temp
                
    import pickle
    pickle.dump( Predictions, open( "%s.p"%spindle_type, "wb" ) )