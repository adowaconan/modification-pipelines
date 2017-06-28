# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 17:13:17 2017
Updated on Mon Apr 10 15:45:33 2017

@author: ning
"""

import eegPinelineDesign
import numpy as np
import mne
import matplotlib.pyplot as plt
plt.rcParams['legend.numpoints']=1
import os
os.chdir('D:\\NING - spindle\\DatabaseSpindles')
import pandas as pd
import random
import seaborn as sns
sns.set_style('white')
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
import pickle
plt.rc('font',size=20,weight='bold')

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
                      front=1,back=1,spindle_segment=3,
                      proba=True):

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
    decision_features=None
    if proba:
        events = mne.make_fixed_length_events(raw,id=1,start=0,duration=spindle_segment)
        epochs = mne.Epochs(raw,events,event_id=1,tmin=0,tmax=spindle_segment,preload=True)
        data = epochs.get_data()[:,:,:-1]
        
        full_prop=[]        
        for d in data:    
            
            rms = eegPinelineDesign.window_rms(d[0,:],moving_window_size)
            l = eegPinelineDesign.trim_mean(rms,0.05) + lower_threshold * eegPinelineDesign.trimmed_std(rms,0.05)
            h = eegPinelineDesign.trim_mean(rms,0.05) + higher_threshold * eegPinelineDesign.trimmed_std(rms,0.05)
            temp_p = (sum(rms>l)+sum(rms<h))/(sum(rms<h) - sum(rms<l))
            if np.isinf(temp_p):
                temp_p = (sum(rms>l)+sum(rms<h))
            full_prop.append(temp_p)
        psds,freq = mne.time_frequency.psd_multitaper(epochs,fmin=11,fmax=16,tmin=0,tmax=3,low_bias=True,)
        psds = 10* np.log10(psds)
        features = pd.DataFrame({'signal':np.array(full_prop),'psd':psds.max(2)[:,0],'freq':freq[np.argmax(psds,2)][:,0]})
        decision_features = StandardScaler().fit_transform(features.values,auto)
        clf = LogisticRegressionCV(Cs=np.logspace(-4,6,11),cv=5,tol=1e-7,max_iter=int(1e7))
        try:
            clf.fit(decision_features,auto)
        except:
            clf.fit(decision_features[:-1],auto)
        auto_proba=clf.predict_proba(decision_features)[:,-1]
    
    auc = roc_auc_score(manual, auto)
    return result,auc,auto,manual,times,auto_proba

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
                    result,auc_,auto,manual,times,auto_proba=new_data_pipeline(raw,spindle_file,
                                                     hypno_file,moving_window_size=100,
                                                    lower_threshold=low,
                                                     higher_threshold=high)
                    all_[str(low)+','+str(high)].append(auc_)
                    t2=time.time()
                    print(t2-t1)
    
                    
    validation_table = {'low':[],'high':[],'AUC_mean':[],'AUC_std':[]}
    for key, item in all_.items():
        low = key.split(',')[0]
        high= key.split(',')[1]
        auc_ = item
        mean_auc=np.mean(auc_)
        std_auc =np.std(auc_)
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
            result,auc_,auto,manual,times,auto_proba = new_data_pipeline(raw,spindle_file,
                                                     hypno_file,moving_window_size=100,
                                                    lower_threshold=best_low,
                                                     higher_threshold=best_high)
            AUC.append(auc_)
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
#    result,_,auto,manual_labels,discritized_times_intervals,auto_proba
    result,_,auto,manual_labels,discritized_times_intervals,auto_proba = new_data_pipeline(raw,spindle_file,hypno_file,moving_window_size=100,
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
#from tpot import TPOTClassifier
print('shuffle')
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
features = data
tpot_data=pd.DataFrame({'class':label},columns=['class'])
############################## model selection #########################
from tpot import TPOTClassifier
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.80)
print('model selection')
tpot = TPOTClassifier(generations=5, population_size=25,
                      verbosity=2,random_state=373849,cv=5,scoring='roc_auc' )
tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)
tpot.export('tpot_exported_pipeline(%d).py'%(1) )  
from sklearn.ensemble import RandomForestClassifier
cv = KFold(n_splits=5,random_state=0,shuffle=True)
fig,ax  = plt.subplots(figsize=(15,15))
for train, test in cv.split(features,label):
    exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(lambda X: X),
        FunctionTransformer(lambda X: X)
    ),
    ExtraTreesClassifier(criterion="entropy", max_features=0.32, n_estimators=500)
    )
#    exported_pipeline = RandomForestClassifier(bootstrap=False, max_features=0.4, min_samples_split=6, n_estimators=100)
    exported_pipeline.fit(features[train],label[train])
    pred_proba=exported_pipeline.predict_proba(features[test])
    fpr,tpr,t= roc_curve(label[test],pred_proba[:,1])
    auc_ = roc_auc_score(label[test],exported_pipeline.predict(features[test]))
    ax.plot(fpr,tpr,label='%.2f'%auc_)
ax.legend(loc='best')
ax.plot([0,1],[0,1],linestyle='--',color='blue')  
#fig.savefig('machine learning (RF).png')  
fig.savefig('machine learning (extratrees).png')  
    
train_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(lambda X: X),
        FunctionTransformer(lambda X: X)
    ),
    ExtraTreesClassifier(criterion="entropy", max_features=0.32, n_estimators=500)
    )
train_pipeline.fit(features,label)
############### individual cross validation ##############
crossvalidation = pd.read_csv('best thresholds.csv')
best_low = crossvalidation['low'].mean()
best_high = crossvalidation['high'].mean()
fif_files = [f for f in os.listdir() if ('fif' in f)]
spindle_files = [f for f in os.listdir() if ('scoring1' in f)]
hypno_files = [f for f in os.listdir() if ('Hypnogram' in f)]
expert2_files = [f for f in os.listdir() if ('scoring2' in f)]
automate_files = [f for f in os.listdir() if ('Automatic' in f)]
idx_files = np.concatenate([np.ones(6),np.zeros(2)])
raws=[]
cv = KFold(n_splits=5,random_state=0,shuffle=True)
for raw_fif in fif_files[:-2]:
    a=mne.io.read_raw_fif(raw_fif,preload=True)
    a.filter(11,16)
    raws.append(a)
all_detection={};all_ML = {};all_expert2={};all_automate={}
from sklearn import metrics
from collections import Counter
from sklearn.model_selection import train_test_split
for raw,spindle_file,hypno_file,expert2_file, auto_file in zip(raws,spindle_files,hypno_files,expert2_files,automate_files):
    result,_,auto_labels,manual_labels,discritized_times_intervals,auto_proba = new_data_pipeline(raw,spindle_file,hypno_file,moving_window_size=100,
                                   lower_threshold=best_low,higher_threshold=best_high)
    temp_auc = [];fp=[];tp=[];confM=[];sensitivity=[];specificity=[]
    if len(auto_proba) > len(manual_labels):
        auto_proba=auto_proba[:-1]
    for train, test in cv.split(manual_labels):
        detected,truth,detected_prob = auto_labels[train],manual_labels[train],auto_proba[train]
        temp_auc.append(roc_auc_score(truth,detected))
        confM_temp = metrics.confusion_matrix(truth,detected)
        TN,FP,FN,TP = confM_temp.flatten()
        sensitivity_ = TP / (TP+FN)
        specificity_ = TN / (TN + FP)
        confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
        confM.append(confM_temp.flatten())
        sensitivity.append(sensitivity_)
        specificity.append(specificity_)
        fp_,tp_,_ = roc_curve(manual_labels[train],auto_proba[train])
        fp.append(fp_);tp.append(tp_)
        print(Counter(truth))
        print(metrics.classification_report(truth,detected))
        print('confusion matrix\n',confM_temp)
#        fpr,tpr,_ = roc_curve(truth,detected)
#        fp.append(fpr);tp.append(tpr)
    
    

    all_detection[raw.filenames[0].split('\\')[-1][:-8]]=[temp_auc,fp,tp,confM,sensitivity,specificity]

    events = mne.make_fixed_length_events(raw,id=1,duration=3)
    epochs = mne.Epochs(raw,events,1,tmin=0,tmax=3,proj=False,preload=True)
    psds, freqs=mne.time_frequency.psd_multitaper(epochs,tmin=0,tmax=3,low_bias=True,proj=False,)
    psds = 10* np.log10(psds)
    data = epochs.get_data()[:,:,:-1];freqs = freqs[psds.argmax(2)];psds = psds.max(2); 
    freqs = freqs.reshape(len(freqs),1,1);psds = psds.reshape(len(psds),1,1)
    data = np.concatenate([data,psds,freqs],axis=2)
    data = data.reshape(len(data),-1)
    labels = manual_labels
    fpr,tpr=[],[];AUC=[];confM=[];sensitivity=[];specificity=[]
    idx_labels = np.arange(len(manual_labels))
    ratio_threshold = list(Counter(manual_labels).values())[1]/list(Counter(manual_labels).values())[0]
    if True:#raw.info['sfreq'] == 200:
        for train, test in cv.split(manual_labels):
#            train,test = train_test_split(idx_labels,random_state=12345)
            exported_pipeline = make_pipeline(
            make_union(
                FunctionTransformer(lambda X: X),
                FunctionTransformer(lambda X: X)
            ),
            ExtraTreesClassifier(criterion="entropy", max_features=0.32, n_estimators=500,
                                 class_weight={1:0.99})
            )
    #        exported_pipeline = RandomForestClassifier(bootstrap=True, 
    #                                                   max_features=0.4, 
    #                                                   min_samples_split=6, 
    #                                                   n_estimators=100,
    #                                                   class_weight={1:0.999,0:0.001})
#            exported_pipeline = train_pipeline
            exported_pipeline.fit(data[train],labels[train])
            fp_,tp_,_ = roc_curve(labels[test],exported_pipeline.predict_proba(data[test])[:,1])
            AUC.append(roc_auc_score(labels[test],
                      exported_pipeline.predict_proba(data[test])[:,-1],average='weighted'))
            fpr.append(fp_);tpr.append(tp_)
            confM_temp = metrics.confusion_matrix(labels[test],
                                                  exported_pipeline.predict_proba(data[test])[:,1]>ratio_threshold)
            TN,FP,FN,TP = confM_temp.flatten()
            sensitivity_ = TP / (TP+FN)
            specificity_ = TN / (TN + FP)
            confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
            confM.append(confM_temp.flatten())
            sensitivity.append(sensitivity_)
            specificity.append(specificity_)
            print(Counter(labels[train]))
            print(metrics.classification_report(labels[test],exported_pipeline.predict(data[test])))
            print('ML,confusion matrix\n',confM_temp)
        all_ML[raw.filenames[0].split('\\')[-1][:-8]]=[AUC,fpr,tpr,confM,sensitivity,specificity]
    else:
        from imblearn.combine import SMOTEENN
        sm = SMOTEENN(random_state=12345)
        data_, labels_ = sm.fit_sample(data,manual_labels)
        for train, test in cv.split(labels_):
            
#            train,test = train_test_split(idx_labels,random_state=12345)
#            sm.fit(data[train], labels[train])
#            train_resampled, label_resampled = sm.sample(data[train], labels[train])
            exported_pipeline = RandomForestClassifier(bootstrap=True, 
                                                       max_features=0.4, 
                                                       min_samples_split=6, 
                                                       n_estimators=100,
                                                       )
            exported_pipeline.fit(data_[train],labels_[train])
#            test_resampled, test_label_resampled = sm.sample(data[test],labels[test])
            fp_,tp_,_ = roc_curve(labels_[test],exported_pipeline.predict_proba(data_[test])[:,1])
            AUC.append(roc_auc_score(labels_[test],
                      exported_pipeline.predict_proba(data_[test])[:,-1]))
            fpr.append(fp_);tpr.append(tp_)
            confM_temp = metrics.confusion_matrix(labels_[test],exported_pipeline.predict(data_[test]))

#            fp_,tp_,_ = roc_curve(labels[test],exported_pipeline.predict_proba(data[test])[:,1])
#            AUC.append(roc_auc_score(labels[test],
#                      exported_pipeline.predict_proba(data[test])[:,-1]))
#            fpr.append(fp_);tpr.append(tp_)
#            confM_temp = metrics.confusion_matrix(labels[test],exported_pipeline.predict(data[test]))
            TN,FP,FN,TP = confM_temp.flatten()
            sensitivity_ = TP / (TP+FN)
            specificity_ = TN / (TN + FP)
            confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
            confM.append(confM_temp.flatten())
            sensitivity.append(sensitivity_)
            specificity.append(specificity_)
            print(Counter(labels_[train]))
            print(metrics.classification_report(labels_[test],exported_pipeline.predict(data_[test])))
            print('ML,confusion matrix\n',confM_temp)
        all_ML[raw.filenames[0].split('\\')[-1][:-8]]=[AUC,fpr,tpr,confM,sensitivity,specificity] 

    expert2_spindle = np.loadtxt(expert2_file,skiprows=1)
    expert2_spindle = pd.DataFrame({'Onset':expert2_spindle[:,0],'Duration':expert2_spindle[:,1]})
    expert2_labels,_ = discritized_onset_label_manual(raw,expert2_spindle,3)
    temp_auc_,fp_,tp_=[],[],[];confM=[];sensitivity=[];specificity=[]
    for train,test in cv.split(manual_labels):
        detected,truth = expert2_labels[train],manual_labels[train]
        temp_auc_.append(roc_auc_score(truth,detected))
        fpr,tpr,_ = roc_curve(truth,detected)
        confM_temp = metrics.confusion_matrix(truth,detected)
        TN,FP,FN,TP = confM_temp.flatten()
        sensitivity_ = TP / (TP+FN)
        specificity_ = TN / (TN + FP)
        confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
        confM.append(confM_temp.flatten())
        sensitivity.append(sensitivity_)
        specificity.append(specificity_)
        fp_.append(fpr);tp_.append(tpr)
        print(Counter(truth))
        print(metrics.classification_report(truth,detected))
        print('confusion matrix\n',confM_temp)
    all_expert2[raw.filenames[0].split('\\')[-1][:-8]]=[temp_auc_,fp_,tp_,confM,sensitivity,specificity]


    automate_spindle = np.loadtxt(auto_file,skiprows=1)
    automate_spindle = pd.DataFrame({'Onset':automate_spindle[:,0],'Duration':automate_spindle[:,1]})
    automate_labes, _ = discritized_onset_label_manual(raw,automate_spindle,3)
    temp_auc__, fp__, tp__=[],[],[];confM=[];sensitivity=[];specificity=[]
    for train, test in cv.split(manual_labels):
        detected, truth = automate_labes[train],manual_labels[train]
        temp_auc__.append(roc_auc_score(truth, detected))
        fpr, tpr, _ = roc_curve(truth, detected)
        fp__.append(fpr);tp__.append(tpr)
        confM_temp = metrics.confusion_matrix(truth,detected)
        TN,FP,FN,TP = confM_temp.flatten()
        sensitivity_ = TP / (TP+FN)
        specificity_ = TN / (TN + FP)
        confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
        confM.append(confM_temp.flatten())
        sensitivity.append(sensitivity_)
        specificity.append(specificity_)
        fp_.append(fpr);tp_.append(tpr)
        print(Counter(truth))
        print(metrics.classification_report(truth,detected))
        print('confusion matrix\n',confM_temp)
    all_automate[raw.filenames[0].split('\\')[-1][:-8]]=[temp_auc__, fp__, tp__,confM,sensitivity,specificity]
all_result = [all_detection,all_ML,all_expert2,all_automate]  
pickle.dump(all_result,open('all cv results.p','wb'))  
import pickle
all_result = pickle.load(open('all cv results.p','rb'))
all_detection,all_ML,all_expert2,all_automate=all_result
def parse_data(all_dict, model='FBT'):
    df_ ={'Subject':[],
              'Mean TN':[],
              'Mean FP':[],
              'Mean FN':[],
              'Mean TP':[],
              'Mean sensitivity':[],
              'Mean specificity':[],
              'Model':[],
              'Mean AUC':[]}
    for keys, values in all_dict.items():
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        mean_confM = np.mean(confM,0)
        std_confM = np.std(confM,0)
        TN,FP,FN,TP = mean_confM
        TN_,FP_,FN_,TP_ = std_confM
        mean_sensitivity = np.mean(sensitivity)
#        std_sensitivity = np.std(sensitivity)
        mean_specificity = np.mean(specificity)
#        std_specificity = np.std(specificity)
        mean_AUC = np.mean(AUC)
#        std_AUC = np.std(AUC)
        df_['Subject'].append(keys)
        df_['Mean TN'].append(TN)
        df_['Mean FP'].append(FP)
        df_['Mean FN'].append(FN)
        df_['Mean TP'].append(TP)
        df_['Mean sensitivity'].append(mean_sensitivity)
        df_['Mean specificity'].append(mean_specificity)
        df_['Model'].append(model)
        df_['Mean AUC'].append(mean_AUC)
    df_ = pd.DataFrame(df_)
    return df_
df_detection = parse_data(all_detection,)
df_ML = parse_data(all_ML,model='Machine learning')
df_expert2 = parse_data(all_expert2,model='Expert 2')
df_automate = parse_data(all_expert2, model='Devuyst et al., 2010')
def define_flash(df_FBT):
    df_FBT['dist'] = abs(df_FBT['Mean AUC'] - df_FBT['Mean AUC'].median())
    flashing_key_FBT = df_FBT['Subject'][np.argmin(df_FBT['dist'])]
    return flashing_key_FBT
flash_detection =define_flash(df_detection)
flash_ML = define_flash(df_ML)
flash_expert2 = define_flash(df_expert2)
flash_automate = define_flash(df_automate)
from scipy import interp
########### plotting ################
##############################################################################
fig= plt.figure(figsize=(24,20));cnt = 0;uv=1.25
ax = fig.add_subplot(131)
xx,yy,xerr,ylabel = [],[],[],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detection.items():
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='blue',
            label='FBT, individual')
ax.axvline(xx.mean(),color='blue',ymax=len(ylabel)/(len(ylabel)+uv),
           )
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='blue',label='FBT average: %.3f $\pm$ %.3f'%(xx.mean(),xx.std()))
sortylabel = [ylabel[ii][-1] for ii in sortIdx ]
_=ax.set(ylim=(-0.5,len(ylabel)+uv),)
plt.xticks(fontsize=16)
plt.yticks(np.arange(len(ylabel)),sortylabel,fontsize=16)
ax.set_ylabel('Subjects',fontsize=20,fontweight='bold')
ax.set_title('Model comparison\ncross validation folds: 5',fontsize=20,fontweight='bold')
ax.set_xlabel('AUC scores ',fontsize=16,fontweight='bold')
xx,yy,xerr,ylabel = [],[],[],[];cnt = 0
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_ML.items():
    yy.append(cnt+0.2)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='red',
            label='Machine learning, individual')

ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           color='red')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='red',label='Machine learning average: %.3f $\pm$ %.3f'%(xx.mean(),xx.std()),)

xx,yy,xerr,ylabel = [],[],[],[];cnt = 0
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_expert2.items():
    yy.append(cnt-0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='green',
            label='Expert 2, individual')

ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           color='green')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='green',label='expert 2 average: %.3f $\pm$ %.3f'%(xx.mean(),xx.std()),)

xx,yy,xerr,ylabel = [],[],[],[];cnt = 0
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_automate.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='black',
            label='Devuyst et al., 2010, individual')

ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           color='black')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='black',label='Devuyst et al., 2010 average: %.3f $\pm$ %.3f'%(xx.mean(),xx.std()),)

#ldg=ax.legend(bbox_to_anchor=(1.5, 0.8))
lgd =ax.legend(loc='upper left',prop={'size':15},frameon=False,scatterpoints=1)
frame = lgd.get_frame()
frame.set_facecolor('None')
######################  machine learning model #########################################
ax_ML = fig.add_subplot(432)
#fig, ax_ML = plt.subplots(figsize=(8,8))
confM_ML = [];cnt=0
for ii, (keys, values) in enumerate(all_ML.items()):
    subject = keys
    if keys == flash_ML:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        confM_ML.append(confM)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)/np.sqrt(5)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        ax_ML.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %s,\nAUC: %.2f $\pm$ %.2f'%(keys[-1],
                                                                                  np.mean(AUC),
                                                                                  np.std(AUC)),
                color='black',alpha=1)
        ax_ML.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                           label='Standard error')
        ax_ML.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        confM_ML.append(confM)
        for auc_n in range(5):
            cnt +=1
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            
            ax_ML.plot(fpr_plot,tpr_plot,color='red',alpha=0.3)
l=ax_ML.legend(loc='lower right',frameon=False,prop={'size':18})
frame = l.get_frame()
frame.set_facecolor('None')
ax_ML.set_title('Between subject ROC AUC\nMachine learning model',fontweight='bold',fontsize=20)
ax_ML.set(ylim=(0,1.02),xlim=(0,1.02))
ax_ML.set_ylabel('True positive rate',fontsize=16,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
########################### FBT model ################################################################
ax_signal = fig.add_subplot(435)
confM_FBT = [];cnt=0
for ii, (keys, values) in enumerate(all_detection.items()):
    subject = keys
    if keys == flash_detection:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        confM_FBT.append(confM)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)/np.sqrt(5)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        ax_signal.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %s,\nAUC: %.2f $\pm$ %.2f'%(keys[-1],
                                                                                  np.mean(AUC),
                                                                                  np.std(AUC)),
                color='black',alpha=1)
        ax_signal.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                           label='Standard error')
        ax_signal.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        for auc_n in range(5):
            confM_FBT.append(confM)
            cnt +=1
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_signal.plot(fpr_plot,tpr_plot,color='blue',alpha=0.3)
ax_signal.legend(loc='lower right',frameon=False,prop={'size':18})
frame = l.get_frame()
frame.set_facecolor('None')
ax_signal.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal.set(ylim=(0,1.02),xlim=(0,1.02))
ax_signal.set_ylabel('True positive rate',fontsize=16,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#ax_signal.set_xlabel('True positive rate',fontsize=15)
###################################### expert 2 ############################################
ax_expert2 = fig.add_subplot(438)
#f, ax_expert2 = plt.subplots()
confM_expert2 = [];cnt=0
for ii, (keys, values) in enumerate(all_expert2.items()):
    subject = keys
    if keys == flash_expert2:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        confM_expert2.append(confM)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)/np.sqrt(5)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        ax_expert2.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %s,\nAUC: %.2f $\pm$ %.2f'%(keys[-1],
                                                                                  np.mean(AUC),
                                                                                  np.std(AUC)),
                color='black',alpha=1)
        ax_expert2.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,)
#                           label='Standard error')
        ax_expert2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        for auc_n in range(5):
            confM_expert2.append(confM)
            cnt +=1
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_expert2.plot(fpr_plot,tpr_plot,color='green',alpha=0.3)
ax_expert2.legend(loc='lower right',frameon=False,prop={'size':18})
frame = l.get_frame()
frame.set_facecolor('None')
ax_expert2.set_title('Expert 2 scoring',fontweight='bold',fontsize=20)
ax_expert2.set(ylim=(0,1.02),xlim=(0,1.02))
ax_expert2.set_ylabel('True positive rate',fontsize=16,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
###################################### the paper model  ############################################
ax_automate = fig.add_subplot(4,3,11)
#f,ax_automate = plt.subplots()
confM_automate = [];cnt=0
for ii, (keys, values) in enumerate(all_automate.items()):
    subject = keys
    if keys == flash_automate:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        confM_automate.append(confM)
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)/np.sqrt(5)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        ax_automate.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %s,\nAUC: %.2f $\pm$ %.2f'%(keys[-1],
                                                                                  np.mean(AUC),
                                                                                  np.std(AUC)),
                color='black',alpha=1)
        ax_automate.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,)
#                           label='Standard error')
        ax_automate.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:
        AUC,fpr,tpr,confM,sensitivity,specificity = values
        for auc_n in range(5):
            confM_automate.append(confM)
            cnt +=1
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_automate.plot(fpr_plot,tpr_plot,color='grey',alpha=0.3)
ax_automate.legend(loc='lower right',frameon=False,prop={'size':18})
frame = l.get_frame()
frame.set_facecolor('None')
ax_automate.set_title('Devuyst et al., 2011 scoring',fontweight='bold',fontsize=20)
ax_automate.set(ylim=(0,1.02),xlim=(0,1.02))
ax_automate.set_xlabel('False positive rate',fontsize=16,fontweight='bold')
ax_automate.set_ylabel('True positive rate',fontsize=16,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
############################### 3rd column ###############################################
confM_FBT_mean = np.array(confM_FBT).mean(1).mean(0)
confM_ML_mean = np.array(confM_ML).mean(1).mean(0)
confM_FBT_std = np.array(confM_FBT).mean(1).std(0)
confM_ML_std = np.array(confM_ML).mean(1).std(0)
confM_expert2_mean = np.array(confM_expert2).mean(1).mean(0)
confM_automate_mean = np.array(confM_automate).mean(1).mean(0)
confM_expert2_std = np.array(confM_expert2).mean(1).std(0)
confM_automate_std = np.array(confM_automate).mean(1).std(0)

ax_ML_CM = fig.add_subplot(433)
ax_ML_CM=sns.heatmap(confM_ML_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_ML_CM,annot=False)
#coors = np.array([[-0.15,0],[1-0.15,0],[-0.15,1],[1-0.15,1]])+ 0.5
coors = np.array([[0,1],[1,1],[0,0],[1,0],])+ 0.5
for ii, (m,s,coor) in enumerate(zip(confM_ML_mean,confM_ML_std,coors)):
    ax_ML_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_ML_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_ML_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_ML_CM.set_title('Between subject confusion matrix\nMachine learning model',fontweight='bold',fontsize=20)
ax_ML_CM.set_ylabel('True label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_signal_CM = fig.add_subplot(436)
ax_signal_CM=sns.heatmap(confM_FBT_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_signal_CM,annot=False)
for ii, (m,s,coor) in enumerate(zip(confM_FBT_mean,confM_FBT_std,coors)):
    ax_signal_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_signal_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_signal_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_signal_CM.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal_CM.set_ylabel('True label',fontsize=20,fontweight='bold')
#ax_signal_CM.set_xlabel('Predicted label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_expert2_CM = fig.add_subplot(439)
ax_expert2_CM = sns.heatmap(confM_expert2_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                            vmin=0,vmax=1.,
                            ax=ax_expert2_CM,annot=False)
for ii, (m,s,coor) in enumerate(zip(confM_expert2_mean,confM_expert2_std,coors)):
    ax_expert2_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_expert2_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_expert2_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_expert2_CM.set_title('Expert 2 scoring',fontweight='bold',fontsize=20)
ax_expert2_CM.set_ylabel('True label',fontsize=16,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_automate_CM = fig.add_subplot(4,3,12)
ax_automate_CM = sns.heatmap(confM_automate_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                            vmin=0,vmax=1.,
                            ax=ax_automate_CM,annot=False)
for ii, (m,s,coor) in enumerate(zip(confM_automate_mean,confM_automate_std,coors)):
    ax_automate_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_automate_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
       xticklabels=['non spindle','spindle'],)
ax_automate_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_automate_CM.set_title('Devuyst et al., 2011 scoring',fontweight='bold',fontsize=20)
ax_automate_CM.set_ylabel('True label',fontsize=16,fontweight='bold')
ax_automate_CM.set_xlabel('Predicted label',fontsize=16,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

fig.tight_layout()
#fig.get_axes()[0].annotate('(a)',(0.45,7.2),fontsize=26)
#fig.get_axes()[1].annotate('(b)',(0.05,1.02),fontsize=26)
#fig.get_axes()[5].annotate('(c)',(0.05,2.002),fontsize=26)
fig.savefig('new data comparison (edit).png')
fig.savefig('new data comparison (high resolution).png',dpi=300)
###################### some statistics #########################################
from random import shuffle
from scipy.stats import percentileofscore
###### permutation test
a = [np.mean(auc_) for keys, (auc_, fpr, tpr,_,_,_) in all_ML.items()]
b = [np.mean(auc_) for keys, (auc_, fpr, tpr,_,_,_) in all_detection.items()]
c = [np.mean(auc_) for keys, (auc_, fpr, tpr,_,_,_) in all_expert2.items()]
d = [np.mean(auc_) for keys, (auc_, fpr, tpr,_,_,_) in all_automate.items()]
pickle.dump([a,b,c,d],open('auc comparison.p','wb'))
temp = pickle.load(open('auc comparison.p','rb'))
a,b,c,d = temp
ps=[]
for tt in range(500):
    diff = []
    difference = np.mean(a) - np.mean(b)
    vector_d = np.concatenate([a,b])
    for ii in range(5000):
        shuffle(vector_d)
        new_a = vector_d[:6]
        new_b = vector_d[6:]
        diff.append(np.mean(new_a) - np.mean(new_b))
    ps.append(min(percentileofscore(diff,difference)/100,(100-percentileofscore(diff,difference))/100))
print('ml vs FBT', np.mean(ps),np.std(ps))    
ps_=[]
for tt in range(500):
    diff = []
    difference = np.mean(a) - np.mean(c)
    vector_d = np.concatenate([a,c])
    for ii in range(5000):
        shuffle(vector_d)
        new_a = vector_d[:6]
        new_c = vector_d[6:]
        diff.append(np.mean(new_a) - np.mean(new_c))
    ps_.append(min(percentileofscore(diff,difference)/100,(100-percentileofscore(diff,difference))/100))
print('ml vs expert2',np.mean(ps_),np.std(ps_))    
ps__=[]
for tt in range(500):
    diff=[]
    difference = np.mean(d) - np.mean(a)
    vector_d = np.concatenate([d,a])
    for ii in range(5000):
        shuffle(vector_d)
        new_d = vector_d[:6]
        new_a = vector_d[6:]
        diff.append(np.mean(new_d) - np.mean(new_a))
    ps__.append(min(percentileofscore(diff,difference)/100,(100-percentileofscore(diff,difference))/100))
print('ml vs paper',np.mean(ps__),np.std(ps__))
ps___=[]
for tt in range(500):
    diff=[]
    difference = np.mean(d) - np.mean(c)
    vector_d = np.concatenate([d,a])
    for ii in range(500):
        shuffle(vector_d)
        new_d = vector_d[:6]
        new_c = vector_d[6:]
        diff.append(np.mean(new_d) - np.mean(new_c))
    ps___.append(min(percentileofscore(diff,difference)/100,(100-percentileofscore(diff,difference))/100))
print('expert2 vs paper',np.mean(ps___),np.std(ps___))

times =pickle.load(open('D:\\NING - spindle\\training set\\step_size_500_11_16getting_higher_threshold\\model running time.p','rb'))
thresholding, machine,_ = times
a, b = np.array(list(thresholding.values())),np.array(list(machine.values()))
pss = []
for tt in range(500):
    diff = []
    difference = np.mean(a) - np.mean(b)
    vector_d = np.concatenate([a,b])
    for ii in range (1000):
        shuffle(vector_d)
        new_a = vector_d[:36]
        new_b = vector_d[36:]
        diff.append(np.mean(new_a) - np.mean(new_b))
    pss.append(min(percentileofscore(diff,difference)/100, (100-percentileofscore(diff,difference))/100))

#####################################################################################
spindle_files = [f for f in os.listdir() if ('scoring1' in f)]
ss = []
for f in spindle_files:
    spindle = np.loadtxt(f,skiprows=1)
    annotation = pd.DataFrame({'Onset':spindle[:,0],'duration':spindle[:,1]})
    ss.append(annotation)
ss = pd.concat(ss)

#######################################################################################
df_all = {'Subject':[],
          'Day':[],
          'Mean TN':[],
          'Mean FP':[],
          'Mean FN':[],
          'Mean TP':[],
          'Mean sensitivity':[],
          'Mean specificity':[],
          'Model':[],
          'Mean AUC':[]}
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detection.items():
    sub = keys
    mean_confM = np.mean(confM,0)
    std_confM = np.std(confM,0)
    TN,FP,FN,TP = mean_confM
    TN_,FP_,FN_,TP_ = std_confM
    mean_sensitivity = np.mean(sensitivity)
    std_sensitivity = np.std(sensitivity)
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    mean_AUC = np.mean(item)
    std_AUC = np.std(item)
    df_all['Subject'].append(sub)
    df_all['Day'].append(np.nan)
    df_all['Mean TN'].append(TN)
    df_all['Mean FP'].append(FP)
    df_all['Mean FN'].append(FN)
    df_all['Mean TP'].append(TP)
    df_all['Mean sensitivity'].append(mean_sensitivity)
    df_all['Mean specificity'].append(mean_specificity)
    df_all['Model'].append('FBT')
    df_all['Mean AUC'].append(mean_AUC) 
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_ML.items():
    sub = keys
    mean_confM = np.mean(confM,0)
    std_confM = np.std(confM,0)
    TN,FP,FN,TP = mean_confM
    TN_,FP_,FN_,TP_ = std_confM
    mean_sensitivity = np.mean(sensitivity)
    std_sensitivity = np.std(sensitivity)
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    mean_AUC = np.mean(item)
    std_AUC = np.std(item)
    df_all['Subject'].append(sub)
    df_all['Day'].append(np.nan)
    df_all['Mean TN'].append(TN)
    df_all['Mean FP'].append(FP)
    df_all['Mean FN'].append(FN)
    df_all['Mean TP'].append(TP)
    df_all['Mean sensitivity'].append(mean_sensitivity)
    df_all['Mean specificity'].append(mean_specificity)
    df_all['Model'].append('ML')
    df_all['Mean AUC'].append(mean_AUC) 
df_all = pd.DataFrame(df_all)   
df_all = df_all.sort_values(['Subject','Day'])
df_all = df_all[['Subject','Day','Mean AUC','Mean TN','Mean FP','Mean FN','Mean TP',
                 'Mean sensitivity','Mean specificity','Model']]
df_all.to_csv('smore measures.csv',index=False)









