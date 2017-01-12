# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:10:59 2017

@author: install
"""

import mne
import eegPinelineDesign
import numpy as np
import pandas as pd
eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
raw = mne.io.read_raw_fif('suj11_l2nap_day2.fif',preload=True)
raw.pick_channels(channelList)
l,h = (11,16)
raw.filter(l,h)
file='suj11_l2nap_day2.fif'

time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_step1(raw,
                                                                                             channelList,file,
                                                                                             moving_window_size=2000,
                                                                                             threshold=.85,
                                                                                             syn_channels=3,
                                                                                             l_freq=l,
                                                                                             h_freq=h,
                                                                                             l_bound=0.55,
                                                                                             h_bound=2.2,
                                                                                              tol=1)
"""Taking out the first 100 seconds and the last 100 seconds"""        
result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
result['Annotation'] = 'auto spindle'
result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
result = result[result.Onset > 100]
annotation_file=['suj11_nap_day2_edited_annotations.txt']
annotation = pd.read_csv(annotation_file[0])
anno = annotation[annotation.Annotation == 'spindle']['Onset']
gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
auto_labels = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
comparedRsult = manual_labels - auto_labels
idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
idx_miss= np.where(comparedRsult == 1)[0]
idx_FA  = np.where(comparedRsult == -1)[0]
discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=3)
discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
discritized_time_intervals = np.array(discritized_time_intervals).T
raw_data, time = raw[:,100*raw.info['sfreq']:-100*raw.info['sfreq']]
def stage_check(x):
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
stages = annotation[annotation.Annotation.apply(stage_check)]
On = stages[::2];Off = stages[1::2]
stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
if abs(np.diff(stage_on_off[0]) - 30) < 2:
    pass
else:
    On = stages[1::2];Off = stages[::2]
    stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
    
for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_miss]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
        temp_data = raw_data[:,idx_start:idx_stop].flatten()
        
for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_FA]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
        temp_data = raw_data[:,idx_start:idx_stop].flatten()
for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_hit]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
        temp_data = raw_data[:,idx_start:idx_stop].flatten()