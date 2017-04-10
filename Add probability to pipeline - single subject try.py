# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 18:15:31 2017

@author: Ning
"""

"""
Improve hard cutting threshold approach -- softmax implementation

Feature? don't know
"""
import numpy as np
import scipy
from scipy.stats import hmean,trim_mean
import mne

import pandas as pd  
from sklearn.metrics import roc_curve,roc_auc_score,auc 
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.preprocessing import StandardScaler 
def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = scipy.signal.gaussian(window_size,(window_size/.68)/2)
  return np.sqrt(np.convolve(a2, window, 'same')/len(a2)) * 1e2
def stage_check(x):
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
def getOverlap(a,b):
    return max(0,min(a[1],b[1]) - max(a[0],b[0]))
def intervalCheck(a,b,tol=0):#a is an array and b is a point
    return a[0]-tol <= b <= a[1]+tol
def trimmed_std(data,percentile):
    temp=data.copy()
    temp.sort()
    percentile = percentile / 2
    low = int(percentile * len(temp))
    high = int((1. - percentile) * len(temp))
    return temp[low:high].std(ddof=0)
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    if spindle_duration_fix:
        spindle_start = spindle - 0.5
        spindle_end   = spindle + 1.5
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
    discritized_continuous_time = np.arange(0,raw.times[-1],step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    temp=[]
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            temp.append([time_interval,spindle])
            if spindle_comparison(time_interval,spindle,spindle_segment):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,temp
def discritized_onset_label_auto(raw,df,spindle_segment):
    spindle_duration = df['Duration'].values
    discritized_continuous_time = np.arange(0,raw.times[-1],step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for kk,spindle in enumerate(df['Onset']):
            if spindle_comparison(time_interval,spindle,spindle_duration[kk],spindle_duration_fix=False):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels,discritized_time_intervals
def thresholding_filterbased_spindle_searching(raw,channelList,annotations,moving_window_size=200,lower_threshold=.9,
                                        syn_channels=3,l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.5,
                                        front=300,back=100,sleep_stage=True,proba=False,validation_windowsize=3):
    
    
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    peak_time={} #preallocate
    sfreq=raw.info['sfreq']
    mph,mpl = {},{}

    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        segment,_ = raw[ii,:]
        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) 
        mph[names] = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS[ii,:],0.05) 
        mpl[names] = trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS[ii,:],0.05)
        pass_ = RMS[ii,:] > mph[names]#should be greater than then mean not the threshold to compute duration

        up = np.where(np.diff(pass_.astype(int))>0)
        down = np.where(np.diff(pass_.astype(int))<0)
        up = up[0]
        down = down[0]
        ###############################
        #print(down[0],up[0])
        if down[0] < up[0]:
            down = down[1:]
        #print(down[0],up[0])
        #############################
        if (up.shape > down.shape) or (up.shape < down.shape):
            size = np.min([up.shape,down.shape])
            up = up[:size]
            down = down[:size]
        C = np.vstack((up,down))
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl[names]:
                    temp_temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time[names].append(temp_temp_time[ints_temp])
                    
        

    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    
    mph['mean'] = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * trimmed_std(RMS_mean,0.05)
    mpl['mean'] = trim_mean(RMS_mean[int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * trimmed_std(RMS_mean,0.05)
    pass_ =RMS_mean > mph['mean']
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    ###############################
    #print(down[0],up[0])
    if down[0] < up[0]:
        down = down[1:]
    #print(down[0],up[0])
    #############################
    if (up.shape > down.shape) or (up.shape < down.shape):
        size = np.min([up.shape,down.shape])
        up = up[:size]
        down = down[:size]
    C = np.vstack((up,down))
    for pairs in C.T:
        
        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
            if np.max(SegmentForPeakSearching)< mpl['mean']:
                temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time['mean'].append(temp_time[ints_temp])
                peak_at.append(SegmentForPeakSearching[ints_temp])
                duration_temp = time[pairs[1]] - time[pairs[0]]
                duration.append(duration_temp) 
            
        
    time_find=[];mean_peak_power=[];Duration=[];
    for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
        temp_timePoint=[]
        for ii, names in enumerate(channelList):
            try:
                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
            except:
                temp_timePoint.append(item + 2)
        try:
            if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
                time_find.append(float(item))
                mean_peak_power.append(PEAK)
                Duration.append(duration_time)
        except:
            pass
    if sleep_stage:
        
        temp_time_find=[];temp_mean_peak_power=[];temp_duration=[];
        # seperate out stage 2
        stages = annotations[annotations.Annotation.apply(stage_check)]
        On = stages[::2];Off = stages[1::2]
        stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
        if abs(np.diff(stage_on_off[0]) - 30) < 2:
            pass
        else:
            On = stages[1::2];Off = stages[::2]
            stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
        for single_time_find, single_mean_peak_power, single_duration in zip(time_find,mean_peak_power,Duration):
            for on_time,off_time in stage_on_off:
                if intervalCheck([on_time,off_time],single_time_find,tol=tol):
                    temp_time_find.append(single_time_find)
                    temp_mean_peak_power.append(single_mean_peak_power)
                    temp_duration.append(single_duration)
        time_find=temp_time_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
    
    result = pd.DataFrame({'Onset':time_find,'Duration':Duration,'Annotation':['spindle']*len(Duration)})     
    auto_label,_ = discritized_onset_label_auto(raw,result,validation_windowsize)
    decision_features=None
    if proba:
        events = mne.make_fixed_length_events(raw,id=1,start=0,duration=validation_windowsize)
        epochs = mne.Epochs(raw,events,event_id=1,tmin=0,tmax=validation_windowsize,preload=True)
        data = epochs.get_data()[:,:,:-1]
        full_prop=[]        
        for d in data:    
            temp_p=[]
            #fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(8,8))
            for ii,(name) in enumerate(zip(channelList)):#,ax.flatten())):
                rms = window_rms(d[ii,:],500)
                l = trim_mean(rms,0.05) + lower_threshold * trimmed_std(rms,0.05)
                h = trim_mean(rms,0.05) + higher_threshold * trimmed_std(rms,0.05)
                prop = (sum(rms>l)+sum(rms<h))/(sum(rms<h) - sum(rms<l))
                temp_p.append(prop)
                
            
            full_prop.append(temp_p)
        psds,freq = mne.time_frequency.psd_multitaper(epochs,fmin=11,fmax=16,tmin=0,tmax=3,low_bias=True,)
        psds = 10* np.log10(psds)
        features = pd.DataFrame(np.concatenate((np.array(full_prop),psds.max(2),freq[np.argmax(psds,2)]),1))
        decision_features = StandardScaler().fit_transform(features.values,auto_label)
        clf = LogisticRegressionCV(Cs=np.logspace(-4,6,11),cv=5,tol=1e-7,max_iter=int(1e7))
        clf.fit(decision_features,auto_label)
        auto_proba=clf.predict_proba(decision_features)[:,-1]
            
    return time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label
    
        
import eegPinelineDesign
eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
raw = mne.io.read_raw_fif('suj11_l2nap_day2.fif',preload=True)
channelList = ['F3','F4','C3','C4','O1','O2']        
raw.pick_channels(channelList)        
raw.filter(11,16)        
   
annotation = pd.read_csv('suj11_nap_day2_edited_annotations.txt')        
time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=thresholding_filterbased_spindle_searching(raw,channelList,annotation,moving_window_size=500,lower_threshold=0.4,higher_threshold=3.4,proba=True)        
       

import re
def spindles(x):
    if re.compile('spindle',re.IGNORECASE).search(x):
        return True
    else:
        return False
spindle = annotation[annotation['Annotation'].apply(spindles)]
manu_label,temp = discritized_onset_label_manual(raw,spindle,3)
      
        
        
        
        
        
        
        
        
        
        
        
        