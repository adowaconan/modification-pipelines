# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:01:01 2017

@author: install
"""

import mne
import eegPinelineDesign
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=22); matplotlib.rc('axes', titlesize=22) 
import numpy as np
import pandas as pd
import scipy.stats as stats
eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData')
channelList = ['F3','F4','C3','C4','O1','O2']
raw = mne.io.read_raw_fif('suj29_l2nap_day2.fif',preload=True)
raw.pick_channels(channelList)
l,h = (11,16);threshold=0.5;hh=3.
raw.filter(l,h)
file='suj29_l2nap_day2.fif'
annotation_file=['D:\\NING - spindle\\allData\\annotation\\suj29_nap_day2_edited_annotations.txt']
annotation = pd.read_csv(annotation_file[0])
time_find,mean_peak_power,Duration,peak_time,peak_at = eegPinelineDesign.spindle_validation_with_sleep_stage(raw,channelList,annotation,moving_window_size=500,threshold=threshold,
                                                                                                             syn_channels=3,l_freq=11,h_freq=16,l_bound=0.5,h_bound=2,tol=1,higher_threshold=hh)

"""
stages = annotation[annotation.Annotation.apply(eegPinelineDesign.stage_check)]
On = stages[::2];Off = stages[1::2]
stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
if abs(np.diff(stage_on_off[0]) - 30) < 2:
    pass
else:
    On = stages[1::2];Off = stages[::2]
    stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
temp_time_find=[];temp_mean_peak_power=[];temp_duration=[];
for single_time_find, single_mean_peak_power, single_duration in zip(time_find,mean_peak_power,Duration):
    for on_time,off_time in stage_on_off:
        if eegPinelineDesign.intervalCheck([on_time,off_time],single_time_find,tol=0):
            temp_time_find.append(single_time_find)
            temp_mean_peak_power.append(single_mean_peak_power)
            temp_duration.append(single_duration)
time_find=temp_time_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
"""
"""Taking out the first 100 seconds and the last 100 seconds"""        
result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
result['Annotation'] = 'auto spindle'
result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
result = result[result.Onset > 100]
Time_ = result.Onset.values
time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
#color=['green','red','blue','black','yellow','orange']
color = plt.cm.viridis(np.linspace(0.4,.5,12))
fig, (ax,ax1,ax2) = plt.subplots(figsize=(20,20),nrows=3,sharex=True)
for ii, names in enumerate(channelList):
    segment,_ = raw[ii,:]
    RMS[ii,:] = eegPinelineDesign.window_rms(segment[0,:],500) 
    ax.plot(time,RMS[ii,:],label=names,alpha=0.3,color=color[ii])
    ax2.plot(time,segment[0,:],label=names,alpha=0.2,color=color[ii])
    mph = eegPinelineDesign.trim_mean(RMS[ii,100000:-30000],0.05) + threshold * eegPinelineDesign.trimmed_std(RMS[ii,:],0.05)
    ax.axhline(mph,color='red',alpha=.3)# higher sd = more strict criteria
    #mpl = eegPinelineDesign.trim_mean(RMS[ii,100000:-30000],0.05) + hh * eegPinelineDesign.trimmed_std(RMS[ii,:],0.05)
    ax.scatter(Time_,np.ones(len(Time_))*mph+0.1*mph,marker='s',color='blue')
                
ax.set(ylabel='RMS',xlim=(time[0],time[-1]))
ax.set_title('Root-mean-square of individual channels',fontsize=22,fontweight='bold')
ax.legend(ncol=6,prop={'size':20})
RMS_mean=stats.hmean(RMS)
mph = eegPinelineDesign.trim_mean(RMS_mean[100000:-30000],0.05) + threshold * eegPinelineDesign.trimmed_std(RMS_mean,0.05)
ax1.plot(time,RMS_mean,color='black',alpha=0.4)
ax1.scatter(Time_,np.ones(len(Time_))*mph+0.1*mph,marker='s',color='blue',alpha=1.)  
ax1.axhline(mph,color='red',alpha=.3)
ax1.set_title('Average root-mean-square of all channels',fontsize=22,fontweight='bold')              
ax1.set(ylabel='RMS')
ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue',alpha=1.)
ax2.set_title('Bandpass EEG signal between 11 - 16 Hz',fontsize=22,fontweight='bold')
ax2.legend(ncol=6,prop={'size':20})
ax2.set(ylabel='$\mu$V',xlabel='Time (Sec)')
anno = annotation[annotation.Annotation == 'spindle']['Onset']
gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,4)# this is step windowsize
auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,4)# this is step windowsize, must be the same
comparedRsult = manual_labels - auto_labels
idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
idx_miss= np.where(comparedRsult == 1)[0]
idx_FA  = np.where(comparedRsult == -1)[0]

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
    ax2.axvspan(time_interval_1,time_interval_2,alpha=0.4,color='red')
for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_FA]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        table.append(pd.DataFrame({'Onset':[time_interval_1],'Annotation':['spindle_FA'],'Duration':[np.nan]}))
    ax2.axvspan(time_interval_1,time_interval_2,alpha=0.4,color='yellow')
for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_hit]):
    if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
        table.append(pd.DataFrame({'Onset':[time_interval_1],'Annotation':['spindle_hit'],'Duration':[np.nan]}))
    ax2.axvspan(time_interval_1,time_interval_2,alpha=0.4,color='black')
for timeStamp in anno.values:
    ax2.annotate('',(timeStamp,-10),(timeStamp,0),arrowprops={'arrowstyle':'<-'},color='k')

fig.tight_layout()
table = pd.concat(table)
table = table[['Onset','Duration','Annotation']]
#table.to_csv('sub11_day2.txt',index=False)


        