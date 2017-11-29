# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 13:55:11 2017

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
from mne.time_frequency import tfr_multitaper,tfr_morlet
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

def get_events(fif,f,validation_windowsize=3,l_threshold=0.4,h_threshold=3.4,channelList=None):

    raw = mne.io.read_raw_fif(fif,preload=True)
    anno = pd.read_csv(f)
    model = Filter_based_and_thresholding()
    if channelList is not None:
        channelList = raw.ch_names[:channelList]
    else:
        channelList = ['F3','F4','C3','C4','O1','O2']
    model.channelList = channelList
    model.get_raw(raw)
    model.get_epochs()
    model.get_annotation(anno)
    model.validation_windowsize = validation_windowsize
    model.syn_channels = int(len(channelList)/2)
#    model.find_onset_duration(l_threshold,h_threshold)
#    model.sleep_stage_check()
    model.make_manuanl_label()
#    model.prepare_validation()
    
    
    #del raw
    return model

exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        )
clf = Pipeline([('scaler',StandardScaler()),
#                ('feature',SelectKBest(f_classif,k=500)),
                ('est',exported_pipeline)])
cv = StratifiedShuffleSplit(n_splits=5,train_size=0.75,test_size=0.25,random_state=12345)


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
annotation = pd.read_csv(f)
picks = mne.pick_types(raw.info,eeg=True,meg=False,eog=False)
raw.filter(11,16,picks=picks)
raw_6 = raw.copy()
raw_32 = raw.copy()
raw_61 = raw.copy()
raw_6.pick_channels(['F3','F4','C3','C4','O1','O2'])
raw_32.pick_channels(raw.ch_names[:32])
raw_61.pick_channels(raw.ch_names[:61])
raw_6.info.normalize_proj()
raw_32.info.normalize_proj()
raw_61.info.normalize_proj()

del raw

#results_6 = eegPinelineDesign.fit_data(raw_6,clf,f,cv,resample=64)
#results_32= eegPinelineDesign.fit_data(raw_32,clf,f,cv,resample=64)
#results_61= eegPinelineDesign.fit_data(raw_61,clf,f,cv,resample=64)
#print(np.mean(results_6[3],0),np.mean(results_32[3],0),np.mean(results_61[3],0))

r6 = eegPinelineDesign.detection_pipeline_crossvalidation(raw_6,raw_6.ch_names,
                                                        annotation,1000,
                                                        0.4,3.4,3,
                                                        11,16,f,cv=cv,
                                                        auc_threshold='adapt')
r32 = eegPinelineDesign.detection_pipeline_crossvalidation(raw_32,raw_32.ch_names,
                                                        annotation,1000,
                                                        0.4,3.4,16,
                                                        11,16,f,cv=cv,
                                                        auc_threshold='adapt')
r61 = eegPinelineDesign.detection_pipeline_crossvalidation(raw_61,raw_61.ch_names,
                                                        annotation,1000,
                                                        0.4,3.4,int(61/2),
                                                        11,16,f,cv=cv,
                                                        auc_threshold='adapt')
print(np.mean(r6[3],0),np.mean(r32[3],0),np.mean(r61[3],0))

front=300;back=100
stop = raw_6.times[-1]-back
events = mne.make_fixed_length_events(raw_6,1,start=front,stop=stop,duration=3,)
epochs = mne.Epochs(raw_6,events,1,tmin=0,tmax=3,proj=False,preload=True)
epochs.resample(64)
gold_standard = eegPinelineDesign.read_annotation(raw_6,f)
manual_labels,_ = eegPinelineDesign.discritized_onset_label_manual(raw_6,gold_standard,3)
freqs = np.arange(11,17,1)
n_cycles = freqs / 2.
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
power = tfr_multitaper(epochs,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)

clf = Pipeline([('vectorizer',Vectorizer()),
                ('scaler',StandardScaler()),
                ('est',exported_pipeline)])

data = power.data
labels = manual_labels
fpr,tpr=[],[];AUC=[];confM=[];sensitivity=[];specificity=[]
for train, test in cv.split(data,labels):
    C = np.array(list(dict(Counter(labels[train])).values()))
    ratio_threshold = C.min() / C.sum()
    print(ratio_threshold)
    clf.fit(data[train,:],labels[train])
    fp,tp,_ = metrics.roc_curve(labels[test],clf.predict_proba(data[test])[:,1])
    confM_temp = metrics.confusion_matrix(labels[test],
                                          clf.predict_proba(data[test])[:,1]>ratio_threshold)
    print('confusion matrix\n',confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis])
    TN,FP,FN,TP = confM_temp.flatten()
    sensitivity_ = TP / (TP+FN)
    specificity_ = TN / (TN + FP)
    AUC.append(metrics.roc_auc_score(labels[test],
              clf.predict_proba(data[test])[:,1]))
    fpr.append(fp);tpr.append(tp)
    confM_temp = confM_temp/ confM_temp.sum(axis=1)[:, np.newaxis]
    confM.append(confM_temp.flatten())
    sensitivity.append(sensitivity_)
    specificity.append(specificity_)
    print(metrics.classification_report(labels[test],
          clf.predict_proba(data[test])[:,1]>ratio_threshold))
print(AUC,confM,sensitivity,specificity)

print(np.mean(results_6[3],0),np.mean(results_32[3],0),np.mean(results_61[3],0),np.mean(confM,0))
#ave = power.data[manual_labels.astype(np.bool)].mean(0).mean(0)
#plt.imshow(ave,origin='lower',aspect='auto',extent=[0,3000,11,16],interpolation='hamming')


















