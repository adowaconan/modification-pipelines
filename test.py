# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:01:01 2017

@author: install
"""

import mne
import eegPinelineDesign
import numpy as np
import pandas as pd
eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData')
channelList = ['F3','F4','C3','C4','O1','O2']
raw = mne.io.read_raw_fif('suj11_l2nap_day2.fif',preload=True)
raw.pick_channels(channelList)
l,h = (11,16)
raw.filter(l,h)
file='suj11_l2nap_day2.fif'
time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at=eegPinelineDesign.get_Onest_Amplitude_Duration_of_spindles(raw,
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
Time_ = result.Onset.values
ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue')                
annotation_file=['annotation\\suj11_nap_day2_edited_annotations.txt']
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
spindle_duration = result['Duration'].values
discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=3)
discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
discritized_time_intervals = np.array(discritized_time_intervals).T

table = []
def stage_check(x):
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
stages = annotation[annotation.Annotation.apply(stage_check)]
On = stages[1::2]
Off = stages[::2]
stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))

for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_miss]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        table.append(pd.DataFrame({'Onset':[time_interval_1],'Annotation':['spindle_miss'],'Duration':[np.nan]}))
    ax2.axvspan(time_interval_1,time_interval_2,alpha=0.1,color='red')
for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_FA]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        table.append(pd.DataFrame({'Onset':[time_interval_1],'Annotation':['spindle_FA'],'Duration':[np.nan]}))
    ax2.axvspan(time_interval_1,time_interval_2,alpha=0.6,color='yellow')
for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_hit]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        table.append(pd.DataFrame({'Onset':[time_interval_1],'Annotation':['spindle_hit'],'Duration':[np.nan]}))
    ax2.axvspan(time_interval_1,time_interval_2,alpha=0.7,color='black')
for timeStamp in anno.values:
    ax2.annotate('',(timeStamp,-100),(timeStamp,0),arrowprops={'arrowstyle':'<-'})

table = pd.concat(table)
table = table[['Onset','Duration','Annotation']]
table.to_csv('sub11_day2.txt',index=False)

        