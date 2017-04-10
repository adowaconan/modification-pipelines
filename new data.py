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
os.chdir('DatabaseSpindles')
import pandas as pd
import random
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc
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
            
            rms = eegPinelineDesign.window_rms(d[0,:],500)
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
#from tpot import TPOTClassifier
print('shuffle')
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
features = data
tpot_data=pd.DataFrame({'class':label},columns=['class'])

#X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.80)
#print('model selection')
#tpot = TPOTClassifier(generations=5, population_size=25,
#                      verbosity=2,random_state=373849,num_cv_folds=5,scoring='roc_auc' )
#tpot.fit(X_train,y_train)
#tpot.score(X_test,y_test)
#tpot.export('tpot_exported_pipeline(%d).py'%(1) )  

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
    exported_pipeline.fit(features[train],label[train])
    pred_proba=exported_pipeline.predict_proba(features[test])
    fpr,tpr,t= roc_curve(label[test],pred_proba[:,1])
    auc = roc_auc_score(label[test],exported_pipeline.predict(features[test]))
    ax.plot(fpr,tpr,label='%.2f'%auc)
ax.legend(loc='best')
ax.plot([0,1],[0,1],linestyle='--',color='blue')  
fig.savefig('machine learning (extratrees).png')  
    

############### individual cross validation ##############
crossvalidation = pd.read_csv('best thresholds.csv')
best_low = crossvalidation['low'].mean()
best_high = crossvalidation['high'].mean()
fif_files = [f for f in os.listdir() if ('fif' in f)]
spindle_files = [f for f in os.listdir() if ('scoring1' in f)]
hypno_files = [f for f in os.listdir() if ('Hypnogram' in f)]
expert2_files = [f for f in os.listdir() if ('scoring2' in f)]
idx_files = np.concatenate([np.ones(6),np.zeros(2)])
raws=[]
cv = KFold(n_splits=5,random_state=0,shuffle=True)
for raw_fif in fif_files[:-2]:
    a=mne.io.read_raw_fif(raw_fif,preload=True)
    a.filter(11,16)
    raws.append(a)
all_detection={};all_ML = {};all_expert2={}
for raw,spindle_file,hypno_file,expert2_file in zip(raws,spindle_files,hypno_files,expert2_files):
    result,_,auto_labels,manual_labels,discritized_times_intervals,auto_proba = new_data_pipeline(raw,spindle_file,hypno_file,moving_window_size=100,
                                   lower_threshold=best_low,higher_threshold=best_high)
    temp_auc = [];fp=[];tp=[]
    
    for train, test in cv.split(manual_labels):
        detected,truth = auto_labels[train],manual_labels[train]
        temp_auc.append(roc_auc_score(truth,detected))
#        fpr,tpr,_ = roc_curve(truth,detected)
#        fp.append(fpr);tp.append(tpr)
    if len(auto_proba) > len(manual_labels):
        auto_proba=auto_proba[:-1]
    fp,tp,_ = roc_curve(manual_labels,auto_proba)

    all_detection[raw.filenames[0].split('\\')[-1][:-8]]=[temp_auc,fp,tp]

    events = mne.make_fixed_length_events(raw,id=1,duration=3)
    epochs = mne.Epochs(raw,events,1,tmin=0,tmax=3,proj=False,preload=True)
    psds, freqs=mne.time_frequency.psd_multitaper(epochs,tmin=0,tmax=3,low_bias=True,proj=False,)
    psds = 10* np.log10(psds)
    data = epochs.get_data()[:,:,:-1];freqs = freqs[psds.argmax(2)];psds = psds.max(2); 
    freqs = freqs.reshape(len(freqs),1,1);psds = psds.reshape(len(psds),1,1)
    data = np.concatenate([data,psds,freqs],axis=2)
    data = data.reshape(len(data),-1)
    labels = manual_labels
    fpr,tpr=[],[];AUC=[]
    for train,test in cv.split(manual_labels):
        exported_pipeline = make_pipeline(
        make_union(
            FunctionTransformer(lambda X: X),
            FunctionTransformer(lambda X: X)
        ),
        ExtraTreesClassifier(criterion="entropy", max_features=0.32, n_estimators=500)
        )
        exported_pipeline.fit(data[train],labels[train])
        fp_,tp_,_ = roc_curve(labels[test],exported_pipeline.predict_proba(data[test])[:,1])
        AUC.append(roc_auc_score(labels[test],
                  exported_pipeline.predict_proba(data[test])[:,-1]))
        fpr.append(fp_);tpr.append(tp_)
    all_ML[raw.filenames[0].split('\\')[-1][:-8]]=[AUC,fpr,tpr]

    expert2_spindle = np.loadtxt(expert2_file,skiprows=1)
    expert2_spindle = pd.DataFrame({'Onset':expert2_spindle[:,0],'Duration':expert2_spindle[:,1]})
    expert2_labels,_ = discritized_onset_label_manual(raw,expert2_spindle,3)
    temp_auc_,fp_,tp_=[],[],[]
    for train,test in cv.split(manual_labels):
        detected,truth = expert2_labels[train],manual_labels[train]
        temp_auc_.append(roc_auc_score(truth,detected))
        fpr,tpr,_ = roc_curve(truth,detected)
        fp_.append(fpr);tp_.append(tpr)
    all_expert2[raw.filenames[0].split('\\')[-1][:-8]]=[temp_auc_,fp_,tp_]
all_result = [all_detection,all_ML,all_expert2]  
pickle.dump(all_result,open('all cv results.p','wb'))  
all_result = pickle.load(open('all cv results.p','rb'))
all_detection,all_ML,all_expert2=all_result

########### plotting ################
fig= plt.figure(figsize=(16,16));cnt = 0;uv=1
ax = fig.add_subplot(121)
xx,yy,xerr,ylabel = [],[],[],[]
for keys, (item,fpr,tpr) in all_detection.items():
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='blue',
            label='Individual performance_thresholding')
ax.axvline(xx.mean(),color='blue',ymax=len(ylabel)/(len(ylabel)+uv),
           label='Thresholding performance: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='blue')
sortylabel = [ylabel[ii] for ii in sortIdx ]
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+uv),
        )
ax.set_title('Individual model comparison results:\ncv=5',fontsize=20,fontweight='bold')
ax.set_xlabel('Area under the curve on predicting spindles and non spindles',fontsize=15)
xx,yy,xerr,ylabel = [],[],[],[];cnt = 0
for keys, (item,fpr,tpr) in all_ML.items():
    yy.append(cnt+0.2)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='red',
            label='Individual performance_ML')

ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           label='Machine learning performance: %.3f'%xx.mean(),color='red')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='red')

xx,yy,xerr,ylabel = [],[],[],[];cnt = 0
for keys, (item,fpr,tpr) in all_expert2.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='green',
            label='Individual performance_expert 2')

ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           label='expert 2 performance: %.3f'%xx.mean(),color='green')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='green')

#ldg=ax.legend(bbox_to_anchor=(1.5, 0.8))
lgd =ax.legend(loc='upper left',prop={'size':12},frameon=False,scatterpoints=1)
frame = lgd.get_frame()
frame.set_facecolor('None')

ax_ML = fig.add_subplot(322)
AUC,fpr,tpr = all_ML['excerpt1']
select = np.random.choice(np.arange(5),size=1)[0]
fpr = fpr[select];tpr = tpr[select]
ax_ML.plot(fpr,tpr,label='Area under the curve: %.3f $\pm$ %.4f'%(np.mean(AUC),np.std(AUC)),color='red')
ax_ML.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
l=ax_ML.legend(loc='best',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_ML.set_title('Machine learning model of \nexcerpt1',fontweight='bold',fontsize=20)
ax_ML.set(ylabel='False positive rate',ylim=(0,1.02))

ax_signal = fig.add_subplot(324)
temp_auc,fp,tp = all_detection['excerpt1']
fp,tp = np.array(fp),np.array(tp)
ax_signal.plot(fp,tp,label='Area under the curve: %.3f $\pm$ %.4f'%(np.mean(temp_auc),np.std(temp_auc)),color='blue')
ax_signal.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_signal.legend(loc='best',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_signal.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal.set(ylabel='False positive rate',ylim=(0,1.02))
#ax_signal.set_xlabel('True positive rate',fontsize=15)

ax_expert2 = fig.add_subplot(326)
temp_auc,fp,tp = all_expert2['excerpt1']
fp,tp = np.array(fp),np.array(tp)
ax_expert2.plot(fp.mean(0),tp.mean(0),label='Area under the curve: %.3f $\pm$ %.4f'%(np.mean(temp_auc),np.std(temp_auc)),color='green')
ax_expert2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_expert2.legend(loc='best',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_expert2.set_title('Expert 2 scoring',fontweight='bold',fontsize=20)
ax_expert2.set(ylabel='False positive rate',ylim=(0,1.02))
ax_expert2.set_xlabel('True positive rate',fontsize=15)

fig.savefig('new data comparison.png')











