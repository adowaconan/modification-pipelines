# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:13:17 2017

@author: ning
"""

import eegPinelineDesign
import numpy as np
import mne
import matplotlib.pyplot as plt
import os
os.chdir('DatabaseSpindles')
import pandas as pd
import random
data_des = pd.read_csv('data description.csv',index_col=False)
def spindle_comparison(times_interval,spindle,spindle_duration,spindle_manual=True):
    if spindle_manual:
        spindle_start = spindle
        spindle_end   = spindle + spindle_duration
        a =  np.logical_or((eegPinelineDesign.intervalCheck(times_interval,spindle_start)),
                           (eegPinelineDesign.intervalCheck(times_interval,spindle_end)))
        return a
    else:
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        a = np.logical_or((eegPinelineDesign.intervalCheck(times_interval,spindle_start)),
                           (eegPinelineDesign.intervalCheck(times_interval,spindle_end)))
        return a
def discritized_onset_label_manual(raw,df,spindle_segment):
    discritized_continuous_times = mne.make_fixed_length_events(raw,id=1,duration=spindle_segment)[:,0]/ raw.info['sfreq']
    discritized_times_intervals = np.vstack((discritized_continuous_times[:-1],discritized_continuous_times[1:]))
    discritized_times_intervals = np.array(discritized_times_intervals).T
    discritized_times_to_zero_one_labels = np.zeros(len(discritized_times_intervals))
    for jj,(times_interval_1,times_interval_2) in enumerate(discritized_times_intervals):
        times_interval = [times_interval_1,times_interval_2]
        for spindle in df['Onset']:
            #print(times_interval,spindle,spindle_segment)
            if spindle_comparison(times_interval,spindle,spindle_segment):
                discritized_times_to_zero_one_labels[jj] = 1
    return discritized_times_to_zero_one_labels,discritized_times_intervals
def discritized_onset_label_auto(raw,df,spindle_segment):
    spindle_duration = df['Duration'].values
    discritized_continuous_times = mne.make_fixed_length_events(raw,id=1,duration=spindle_segment)[:,0]/ raw.info['sfreq']
    discritized_times_intervals = np.vstack((discritized_continuous_times[:-1],discritized_continuous_times[1:]))
    discritized_times_intervals = np.array(discritized_times_intervals).T
    discritized_times_to_zero_one_labels = np.zeros(len(discritized_times_intervals))
    for jj,(times_interval_1,times_interval_2) in enumerate(discritized_times_intervals):
        times_interval = [times_interval_1,times_interval_2]
        for kk,spindle in enumerate(df['Onset']):
            if spindle_comparison(times_interval,spindle,spindle_duration[kk],spindle_manual=False):
                discritized_times_to_zero_one_labels[jj] = 1
    return discritized_times_to_zero_one_labels,discritized_times_intervals
def new_data_pipeline(raw,annotation_file_name,Hypnogram_file_name,
                      moving_window_size=200,
                      lower_threshold=.9,higher_threshold=3.5,
                      l_freq=11,h_freq=16,
                      l_bound=0.5,h_bound=2,tol=1,
                      front=1,back=1,spindle_segment=3):

    # prepare annotations   
    spindle = np.loadtxt(annotation_file_name,skiprows=1)
    annotation = pd.DataFrame({'Onset':spindle[:,0],'duration':spindle[:,1]})
    
    sleep_stage = np.loadtxt(Hypnogram_file_name,skiprows=1)
    events = mne.make_fixed_length_events(raw,id=1,
                                          duration=round(raw.last_samp/raw.info['sfreq']/len(sleep_stage)))
    stage_on_off=pd.DataFrame(np.array([events[:-1,0]/raw.info['sfreq'],
                                    events[1:,0]/raw.info['sfreq'],sleep_stage[:-1]]).T,columns=['on','off','stage'])
    stage_on_off = stage_on_off[stage_on_off['stage']==2.]
    
    # signal processing and thresholding
    #raw.filter(l_freq,h_freq)
    sfreq = raw.info['sfreq']
    times=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(raw.ch_names),raw._data[0,:].shape[0]))
    #preallocate
    for ii, names in enumerate(raw.ch_names):
        segment,_ = raw[ii,:]
        RMS[ii,:] = eegPinelineDesign.window_rms(segment[0,:],moving_window_size) # window of 200ms
        mph = eegPinelineDesign.trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + lower_threshold * eegPinelineDesign.trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
        mpl = eegPinelineDesign.trim_mean(RMS[ii,int(front*sfreq):-int(back*sfreq)],0.05) + higher_threshold * eegPinelineDesign.trimmed_std(RMS[ii,:],0.05)
        pass_ = RMS[ii,:] > mph#should be greater than then mean not the threshold to compute duration

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
        peak_=[];duration=[];peak_times=[]
        for pairs in C.T:
            if l_bound < (times[pairs[1]] - times[pairs[0]]) < h_bound:
                timesPoint = np.mean([times[pairs[1]],times[pairs[0]]])
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl:
                    temp_temp_times = times[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_times.append(temp_temp_times[ints_temp])
                    peak_.append(SegmentForPeakSearching[ints_temp])
                    duration_temp = times[pairs[1]] - times[pairs[0]]
                    duration.append(duration_temp)
    temp_times_find=[];temp_mean_peak_power=[];temp_duration=[];
    for single_times_find, single_mean_peak_power, single_duration in zip(peak_times,peak_,duration):
        for ii,row in stage_on_off.iterrows():
            on_times,off_times = row['on'],row['off']
            #print(on_times,off_times,single_times_find)
            if eegPinelineDesign.intervalCheck([on_times,off_times],single_times_find,tol=tol):
                temp_times_find.append(single_times_find)
                temp_mean_peak_power.append(single_mean_peak_power)
                temp_duration.append(single_duration)
    times_find=temp_times_find;mean_peak_power=temp_mean_peak_power;Duration=temp_duration
    result = pd.DataFrame({'Onset':times_find,'Peak':mean_peak_power,
                          'Duration':Duration})
    
    # validation
    manual,_ = discritized_onset_label_manual(raw,annotation,spindle_segment)
    auto,times= discritized_onset_label_auto(raw,result,spindle_segment)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(manual, auto)
    return result,auc,auto,manual,times

fif_files = [f for f in os.listdir() if ('fif' in f)]
spindle_files = [f for f in os.listdir() if ('scoring1' in f)]
hypno_files = [f for f in os.listdir() if ('Hypnogram' in f)]
idx_files = np.concatenate([np.ones(6),np.zeros(2)])
raws=[]
for raw_fif in fif_files:
    a=mne.io.read_raw_fif(raw_fif,preload=True)
    a.filter(11,16)
    raws.append(a)
crossvalidation={'low':[],'high':[],'AUC':[]}
import time
for aa in range(10):
    all_ = {}
    for low in np.arange(0.1,.9,0.1):
        for high in np.arange(2.5,3.6,0.1):
            all_[str(low)+','+str(high)]=[]
    for ii in range(100):
        random.shuffle(idx_files)
    print(idx_files)
    for low in np.arange(0.1,.9,0.1):
        for high in np.arange(2.5,3.6,0.1):
            print(aa,low,high)
            for raw,spindle_file,hypno_file,train in zip(raws,spindle_files,hypno_files,idx_files):
                if train:
                    t1=time.time()
                    result,auc,auto,_,_=new_data_pipeline(raw,spindle_file,
                                                     hypno_file,moving_window_size=100,
                                                    lower_threshold=low,
                                                     higher_threshold=high)
                    all_[str(low)+','+str(high)].append(auc)
                    t2=time.time()
                    print(t2-t1)
    
                    
    validation_table = {'low':[],'high':[],'AUC_mean':[],'AUC_std':[]}
    for key, item in all_.items():
        low = key.split(',')[0]
        high= key.split(',')[1]
        auc = item
        mean_auc=np.mean(auc)
        std_auc =np.std(auc)
        validation_table['low'].append(low)
        validation_table['high'].append(high)
        validation_table['AUC_mean'].append(mean_auc)
        validation_table['AUC_std'].append(std_auc)
    validation_table = pd.DataFrame(validation_table)
    best = validation_table[validation_table['AUC_mean'] == validation_table['AUC_mean'].max()]
    best_low = float(best['low'].values[0]);best_high=float(best['high'].values[0])
    test = 1-idx_files
    AUC=[]
    for raw,spindle_file,hypno_file,train in zip(raws,spindle_files,hypno_files,test):
        if train:
            result,auc,auto,manual,times = new_data_pipeline(raw,spindle_file,
                                                     hypno_file,moving_window_size=100,
                                                    lower_threshold=best_low,
                                                     higher_threshold=best_high)
            AUC.append(auc)
    crossvalidation['low'].append(best_low)
    crossvalidation['high'].append(best_high)
    crossvalidation['AUC'].append(AUC)
crossvalidation = pd.DataFrame(crossvalidation)
crossvalidation.to_csv('best thresholds.csv',index=False)

crossvalidation = pd.read_csv('best thresholds.csv')
best_low = crossvalidation['low'].mean()
best_high = crossvalidation['high'].mean()
# collect sample to train machine learning models
def sampling_FA_MISS_CR(comparedRsult,manual_labels, raw , discritized_times_intervals,sample,label):
    if raw.info['sfreq'] == 200:
        idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
        idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
        idx_miss= np.where(comparedRsult == 1)[0]
        idx_FA  = np.where(comparedRsult == -1)[0]
        #sleep_stage = np.loadtxt(Hypnogram_file_name,skiprows=1)
        events = mne.make_fixed_length_events(raw,id=1,duration=3)
    #    stage_on_off=pd.DataFrame(np.array([events[:-1,0]/raw.info['sfreq'],
    #                                    events[1:,0]/raw.info['sfreq'],sleep_stage[:-1]]).T,columns=['on','off','stage'])
    #    stage_on_off = stage_on_off[stage_on_off['stage']==2.]
        epochs = mne.Epochs(raw,events,1,tmin=0,tmax=3,proj=False,preload=True)
        psds, freqs=mne.time_frequency.psd_multitaper(epochs,tmin=0,tmax=3,low_bias=True,proj=False,)
        psds = 10* np.log10(psds)
        data = epochs.get_data()[:,:,:-1];freqs = freqs[psds.argmax(2)];psds = psds.max(2); 
        freqs = freqs.reshape(len(freqs),1,1);psds = psds.reshape(len(psds),1,1)
        data = np.concatenate([data,psds,freqs],axis=2)
        data = data.reshape(len(data),-1)
        sample.append(data[idx_miss]);label.append(np.ones(len(idx_miss)))
        sample.append(data[idx_FA]);label.append(np.zeros(len(idx_FA)))
        
        len_need = len(idx_FA) - len(idx_miss)
        if len_need > 0:
            try:
                idx_hit_need = np.random.choice(idx_hit,size=len_need,replace=False)
            except:
                idx_hit_need = np.random.choice(idx_hit,size=len_need,replace=True)
            sample.append(data[idx_hit_need])
            label.append(np.ones(len(idx_hit_need)))
        else:
            idx_CR_nedd = np.random.choice(idx_CR,len_need,replace=False)
            sample.append(data[idx_CR_nedd])
            label.append(np.zeros(len(idx_CR_nedd)))
        return sample,label
    else:
        return sample,label
sample=[];label=[];import pickle
for raw,spindle_file,hypno_file in zip(raws,spindle_files,hypno_files):
    result,_,auto,manual_labels,discritized_times_intervals = new_data_pipeline(raw,spindle_file,hypno_file,moving_window_size=100,
                                   lower_threshold=best_low,higher_threshold=best_high)
    #raw = mne.io.read_raw_fif(raw_file,preload=True)
    comparedRsult = manual - auto
    sample,label=sampling_FA_MISS_CR(comparedRsult,manual_labels, raw , discritized_times_intervals,sample,label)
    pickle.dump(sample,open('sample_data.p','wb'))
    pickle.dump(label,open('smaple_label.p','wb'))
    
sample = pickle.load(open('sample_data.p','rb'))
label = pickle.load(open('smaple_label.p','rb'))
data,label = np.concatenate(sample),np.concatenate(label)
idx_row=np.arange(0,len(label),1)
from random import shuffle
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
print('shuffle')
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
features = data
tpot_data=pd.DataFrame({'class':label},columns=['class'])
for aa in range(10):
    X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.80)
    print('model selection')
    tpot = TPOTClassifier(generations=10, population_size=25,
                          verbosity=2,random_state=373849,num_cv_folds=5,scoring='roc_auc' )
    tpot.fit(X_train,y_train)
    tpot.score(X_test,y_test)
    tpot.export('tpot_exported_pipeline(%d).py'%(aa+1) )  