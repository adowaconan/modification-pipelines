# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:42:51 2016

@author: ning
"""

import eegPinelineDesign
import numpy as np
import random
import mne
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import re
import json
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA,FastICA
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from scipy.fftpack import fft,ifft
import math
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from scipy.signal import spectrogram,find_peaks_cwt,butter, lfilter
from mne.preprocessing.ica import ICA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split,ShuffleSplit
from sklearn.preprocessing import label_binarize,scale
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize,StandardScaler


eegPinelineDesign.change_file_directory('C:/Users/ning/Downloads/training set')
EDFfiles, Annotationfiles = eegPinelineDesign.split_type_of_files()
from eegPinelineDesign import CenterAtPeakOfWindow,Threshold_test,spindle_overlapping_test,used_windows_check,cut_segments

import pickle
with open('training data.p','rb') as handle:
    result=pickle.load(handle)
    
channelList=['F3','F4','C3','C4','O1','O2']
X = [];Y=[]
for num, EDFfileName in enumerate(EDFfiles):
    if EDFfileName == 'suj13_l2nap_day2 edited.edf' or EDFfileName =='suj13_l2nap_day2 edited1.edf':
        pass
    else:
        file_to_read,fileName=eegPinelineDesign.pick_sample_file(EDFfiles,n=num)


        for names in channelList:
            for item in result['spindle'][fileName][names]:
                if item.shape[1] == 3000:

                    X.append(item[0,:])
                    Y.append(1)
            for item in result['non spindle'][fileName][names]:
                if item.shape[1] == 3000:
                    X.append(item[0,:])
                    Y.append(0)

X=np.array(X);Y=np.array(Y)

idx=np.arange(X.shape[0])
GG=np.random.choice(tuple(idx),len(idx),replace=False)
def shuffle(x):
    return sorted(x, key=lambda k: random.random())
GG = shuffle(shuffle(shuffle(shuffle(GG))))
XX=[];YY=[]
for idxx in GG:
    XX.append(X[idxx])
    YY.append(Y[idxx])
    
from sklearn.preprocessing import normalize
normal_X = normalize(XX)

clf =LogisticRegression(penalty='l2',C=.1,tol=10e-9,fit_intercept=True,solver='liblinear',
                                             max_iter=5e8,multi_class='ovr',n_jobs=-1)



clf.fit(normal_X,YY)


num=2
file_to_read,fileName=eegPinelineDesign.pick_sample_file(EDFfiles,n=num)
channelList = ['F3','F4','C3','C4','O1','O2','ROC','LOC']
raw_filter = eegPinelineDesign.load_data(file_to_read,channelList,11, 16)#spindle pass
raw_alpha=eegPinelineDesign.load_data(file_to_read,channelList,8, 12)#alpha pass
raw_spindle=eegPinelineDesign.load_data(file_to_read,channelList,11, 16)#spindle pass
raw_muscle=eegPinelineDesign.load_data(file_to_read,channelList,30, 40)#
channelList = ['F3','F4','C3','C4','O1','O2']
raw_filter.pick_channels(channelList)


TimePoint = 0+5;cnt = 0
time_label={};resolution = 1
for names in channelList:
    time_label[names]=[]
while raw_filter.last_samp/1000 - TimePoint > 5:
    if any(abs(np.array(spindles) - TimePoint) < 1):
        print(TimePoint,np.array(spindles)[np.argmin(abs(np.array(spindles)-TimePoint))])
    else:
        print(TimePoint)
    for ii,names in enumerate(channelList):
        if Threshold_test(TimePoint,raw_alpha,raw_spindle,raw_muscle,ii):
            tempSegment,timeSpan=cut_segments(raw_filter,TimePoint,ii,windowsize = 1.5)
            normalziedSegment = normalize(tempSegment[0,:3000])#key step!!!
            predictedLabel = clf.predict(normalziedSegment)
            if predictedLabel == 1:
                # only find the peak of spindles, if not, why bother
                peakXval = eegPinelineDesign.CenterAtPeakOfWindow(TimePoint,1.5,raw_filter,ii)
                time_label[names].append([[TimePoint-1.5,TimePoint+1.5],peakXval,predictedLabel])
                print([TimePoint-1.5,TimePoint+1.5],peakXval,predictedLabel)
            else:
                time_label[names].append([[TimePoint-1.5, TimePoint+1.5],TimePoint,predictedLabel])
                print([TimePoint-1.5,TimePoint+1.5],TimePoint,predictedLabel)
        else:
            time_label[names].append([[TimePoint-1.5, TimePoint+1.5],TimePoint,0])
            print([TimePoint-1.5,TimePoint+1.5],TimePoint,0)
    TimePoint += resolution
