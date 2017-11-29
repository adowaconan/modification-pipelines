# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:37:16 2017

@author: ning
"""

import mne
from tqdm import tqdm
from collections import Counter

from sklearn.pipeline import make_pipeline,make_union,Pipeline
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_predict
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn import metrics

import os
os.chdir('D:/Ning - spindle/')
import eegPinelineDesign
from eegPinelineDesign import getOverlap#thresholding_filterbased_spindle_searching
from Filter_based_and_thresholding import Filter_based_and_thresholding

from matplotlib import pyplot as plt
from scipy import stats
from mne.time_frequency import tfr_multitaper,tfr_array_multitaper
from mne.decoding import Vectorizer

import pandas as pd
import re
import numpy as np
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
annotations = [f for f in os.listdir() if ('annotations.txt' in f)] # get all the possible annotation files
fif_data = [f for f in os.listdir() if ('raw_ssp.fif' in f)] # get all the possible preprocessed data, might be more than or less than annotation files
def spindle(x,KEY='spindle'):# iterating through each row of a data frame and matcing the string with "KEY"
    keyword = re.compile(KEY,re.IGNORECASE) # make the keyword
    return keyword.search(x) != None # return true if find a match
    
exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        )
clf = Pipeline([('scaler',StandardScaler()),
#                ('feature',SelectKBest(f_classif,k=500)),
                ('est',exported_pipeline)])
cv = StratifiedShuffleSplit(n_splits=5,train_size=0.75,test_size=0.25,random_state=12345)
freqs = np.arange(11,17,1)
n_cycles = freqs / 2.
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)

f = annotations[37]
temp_ = re.findall('\d+',f)
sub = temp_[0] # the first one will always be subject number
day = temp_[-1]# the last one will always be the day

if int(sub) < 11: # change a little bit for matching between annotation and raw EEG files
    day = 'd%s' % day
else:
    day = 'day%s' % day
fif_file = [f for f in fif_data if ('suj%s_'%sub in f.lower()) and (day in f)][0]# the .lower() to make sure the consistence of file name cases
print(sub,day,f,fif_file) # a checking print 

raw = mne.io.read_raw_fif(fif_file,preload=True)
anno = pd.read_csv(f)
model = Filter_based_and_thresholding()
channelList = 32
if channelList == None:
    channelList = ['F3','F4','C3','C4','O1','O2']
else:
    channelList = raw.ch_names[:32]
model.channelList = channelList
model.get_raw(raw)
model.get_epochs(resample=64)
model.get_annotation(anno)
model.validation_windowsize = 3
model.syn_channels = int(len(channelList)/2)
model.find_onset_duration(0.4,3.4)
model.sleep_stage_check()
model.make_manuanl_label()
event_interval = model.epochs.events[:,0] / 1000
event_interval = np.vstack([event_interval, event_interval + 3]).T
sleep_stage_interval = np.array(model.stage_on_off)
row_idx = [sum([getOverlap(interval,temp) for temp in sleep_stage_interval]) != 0 for interval in event_interval]
labels = model.manual_labels[row_idx]

data = model.epochs.get_data()[row_idx]
info = model.epochs.info
events = model.epochs.events[row_idx]
events[:,0] = events[:,0] / model.epochs.info['sfreq']
events[:,-1] = labels
event_id = {'spindle':1,'non spindle':0}
epochs_ = mne.EpochsArray(data,info,events=events,tmin=0,event_id=event_id,)
power = tfr_multitaper(model.epochs,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
data = power.data


power = tfr_multitaper(epochs_,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
data_ = power.data

labels_ = epochs_.events[:,-1]
labels = model.manual_labels
clf = Pipeline([('vectorizer',Vectorizer()),
                ('scaler',StandardScaler()),
                ('est',exported_pipeline)])
#fpr,tpr=[],[];AUC=[];confM=[];sensitivity=[];specificity=[]
#for train, test in cv.split(data_,labels_):
#    C = np.array(list(dict(Counter(labels_[train])).values()))
#    ratio_threshold = C.min() / C.sum()
#    print(ratio_threshold)
#    clf.fit(data_[train,:],labels_[train])
#    pred = clf.predict(data)
#    pred_prob = clf.predict_proba(data)[:,1]
#    fp,tp,_ = metrics.roc_curve(labels,pred_prob)
#    confM_temp = metrics.confusion_matrix(labels,pred_prob>ratio_threshold)
#    print('confusion matrix\n',confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis])
#    TN,FP,FN,TP = confM_temp.flatten()
#    sensitivity_ = TP / (TP+FN)
#    specificity_ = TN / (TN + FP)
#    AUC.append(metrics.roc_auc_score(labels,pred_prob))
#    fpr.append(fp);tpr.append(tp)
#    confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
#    confM.append(confM_temp.flatten())
#    sensitivity.append(sensitivity_)
#    specificity.append(specificity_)
#    print(metrics.classification_report(labels,pred_prob>ratio_threshold))

clf.fit(data_,labels_)
pred_,pred_prob_ = [],[]
n_  = np.array([0,5000])

while model.raw.times[-1] * 1000 - n_ [1]> 0:
    idx = model.raw.times[n_[0]:n_[1]] 
    temp_event = pd.DataFrame(idx.reshape(5000,1) * 1000 ,columns=['onset'])
    temp_event['i'] = 0
    temp_event['code'] = 1
    temp_event = temp_event.values.astype(int)
    temp_epoch = mne.Epochs(raw,temp_event,event_id=1,tmin=0,tmax=3,preload=True,)
    temp_epoch.resample(64)
    temp_power = tfr_multitaper(temp_epoch,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
    temp_data = temp_power.data
    pred_.append(clf.predict(temp_data))
    pred_prob_.append(clf.predict_proba(temp_data)[:,-1])
    
    
    
    n_ += 1000

for ii,onset in enumerate(model.raw.times):
    offset = onset + 3
    temp_event = pd.DataFrame(np.array([onset,0,1]).reshape(1,3),columns=['onset','i','code'])
    temp_event = temp_event.values.astype(int)
    temp_epoch = mne.Epochs(raw,temp_event,event_id=1,tmin=0,tmax=3,preload=True,)
    temp_epoch.resample(64)
    temp_power = tfr_multitaper(temp_epoch,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
    temp_data = temp_power.data
    pred_.append(clf.predict(temp_data))
    pred_prob_.append(clf.predict_proba(temp_data))
    















