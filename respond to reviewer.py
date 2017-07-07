# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:52:09 2017

@author: ning
"""
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import eegPinelineDesign
from scipy.stats import trim_mean
plt.rc('axes', titlesize=30)
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
import warnings
warnings.filterwarnings("ignore")

file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=500;threshold=0.4;syn_channel=3
l,h = (11,16);
low, high=11,16
hh=3.4
front=300;back=100;total=front+back
def find_peaks(raw,channelList,windowSize,threshold,hh,result):         
    ms = np.zeros([6,2])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    idx_left = [0,2,4]
    idx_right = [1,3,5]
    for ii,name in enumerate(channelList):
        segment,_ = raw[ii,:]
        RMS[ii,:] = eegPinelineDesign.window_rms(segment[0,:],windowSize) 
        mph = trim_mean(RMS[ii,100000:-30000],0.05) + threshold * eegPinelineDesign.trimmed_std(RMS[ii,100000:-30000],0.05) 
        mpl = trim_mean(RMS[ii,100000:-30000],0.05) + hh * eegPinelineDesign.trimmed_std(RMS[ii,100000:-30000],0.05)
        ms[ii,:] = [mph, mpl]
    peaks = []
    for time_stamp,duration in zip(result.Onset,result.Duration):
        start, stop = time_stamp - duration, time_stamp + duration
        start_, stop_ = raw.time_as_index([start,stop])
        segment,times = raw[:,start_:stop_]
        temp = []
        for ii, name in enumerate(channelList):
            info = mne.create_info([name],raw.info['sfreq'])
            E = mne.EvokedArray(segment[ii,:].reshape(1,-1),info)
            _,peak = E.get_peak(mode='pos')
            
            temporal_mark = np.argmin(abs(times-(peak+start)<0.01))
            temp.append(segment[ii,temporal_mark])
        peaks.append(temp)
    peaks = np.array(peaks)
    #peaks = peaks / peaks.std(0)
    return peaks[:,idx_left], peaks[:,idx_right]
results = []
comparison = []
if True:
    for file in np.random.choice(list_file_to_read,size=len(list_file_to_read),replace=False):
        sub = file.split('_')[0]
        if int(sub[3:]) >= 11:
            day = file.split('_')[2][:4]
            day_for_show = day
            old = False
        else:
            day = file.split('_')[1]
            day_for_show = day[0]+'ay'+day[1]
            old = True
    
        annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
        if len(annotation_file) != 0:
            annotations = pd.read_csv(annotation_file[0])
            raw = mne.io.read_raw_fif(file,preload=True)
            if old:
                pass
            else:
                raw.resample(500, npad="auto") # down sampling Karen's data
            raw.pick_channels(channelList)
            raw.filter(low,high)
            time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
                                                                                                    channelList,annotations,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=threshold,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=hh,
                                                                                                    front=300,back=100,sleep_stage=True,)
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
            result = result[result.Onset > front]
            temp_left, temp_right = find_peaks(raw,channelList,windowSize,threshold,hh,result)
            lefts = np.array(temp_left).flatten()
            rights = np.array(temp_right).flatten()
            comparison.append(lefts.mean() - rights.mean())
            results.append([temp_left,temp_right])
results = np.concatenate(results,axis=1)
left = results[0].flatten()
right = results[1].flatten()

fig,ax = plt.subplots(figsize=(16,8),ncols=2)
ax[0].hist(left,bins=50,color='red',alpha=0.6,label='F3,C3,O1',normed=True)
ax[0].hist(right,bins=50,color='blue',alpha=0.6,label='F4,C4,O2',normed=True)
ax[0].set(xlabel='$\mu$V',ylabel='Normalized hist',title='Between-subject (N=%d)'%(len(left)))
ax[0].legend()
ax[1].bar([0,1],[sum(np.array(comparison)>0),sum(np.array(comparison)<0)])
ax[1].set(xticks=[0,1],xticklabels=['left>right','left<right'],title='Within-subject (N=%d)'%(len(comparison)),
  ylabel='Counts')
fig.savefig('D:\\NING - spindle\\training set\\left vs right channels.png',dpi=300)

from random import shuffle
from scipy.stats import percentileofscore

def Permutation_test(data1, data2, n1=100,n2=100):
    p_values = []
    for simulation_time in range(n1):
        shuffle_difference =[]
        experiment_difference = np.mean(data1,0) - np.mean(data2,0)
        vector_concat = np.concatenate([data1,data2])
        for shuffle_time in range(n2):
            shuffle(vector_concat)
            new_data1 = vector_concat[:len(data1)]
            new_data2 = vector_concat[len(data1):]
            shuffle_difference.append(np.mean(new_data1) - np.mean(new_data2))
        p_values.append(min(percentileofscore(shuffle_difference,experiment_difference)/100,
                            (100-percentileofscore(shuffle_difference,experiment_difference))/100))
    
    return p_values,np.mean(p_values),np.std(p_values)
a,b,c = Permutation_test(left,right)
