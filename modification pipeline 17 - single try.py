# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:32:06 2016

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
try:
    eegPinelineDesign.change_file_directory('C:/Users/ning/Downloads/training set')
except:
    pass
EDFfiles, Annotationfiles = eegPinelineDesign.split_type_of_files()

from eegPinelineDesign import CenterAtPeakOfWindow,Threshold_test,spindle_overlapping_test,used_windows_check,cut_segments


num=int(input("please enter 1 or 2"))
spindle={};nonspindle={};time_spindle={};time_nonspindle={}
file_to_read,fileName=eegPinelineDesign.pick_sample_file(EDFfiles,n=num)
channelList = ['F3','F4','C3','C4','O1','O2','ROC','LOC']
#raw = eegPinelineDesign.load_data(file_to_read,channelList,None, 100)#low pass data
raw_alpha=eegPinelineDesign.load_data(file_to_read,channelList,8, 12)#alpha pass
raw_spindle=eegPinelineDesign.load_data(file_to_read,channelList,11, 16)#spindle pass
raw_muscle=eegPinelineDesign.load_data(file_to_read,channelList,30, 40)#
annotation_to_read = [x for x in Annotationfiles if fileName in x]
file = pd.read_csv(annotation_to_read[0])
labelFind = re.compile('spindle',eegPinelineDesign.re.IGNORECASE)
spindles=[]# take existed annotations
for row in file.iterrows():
    currentEvent = row[1][-1]
    if labelFind.search(currentEvent):
        spindles.append(row[1][0])# time of marker
print('finish spindle annotation')

channelList=['F3','F4','C3','C4','O1','O2']
plt.figure(figsize=(20,20))
used_time_windows=[]
time_spindle[fileName]={};time_nonspindle[fileName]={}
spindle[fileName]={};nonspindle[fileName]={}
for channelID, names in enumerate(channelList):
    time_spindle[fileName][names]=[];time_nonspindle[fileName][names]=[]
    spindle[fileName][names]=[];nonspindle[fileName][names]=[]
    print(names)
    used_time_windows=[]
    for items in spindles:
        if Threshold_test(items,raw_alpha,raw_spindle,raw_muscle,channelID):
            try:
                centered_marker=CenterAtPeakOfWindow(items,2,raw_spindle,channelID)
                Segment,TT = cut_segments(raw_spindle,centered_marker,channelID)
                spindle[fileName][names].append(Segment)
                print(fileName,names,centered_marker,'spindle')
                #plt.plot(TT,Segment)
                #plt.clf()
                for max_iteration in range(1000):
                    startPoint=6;endPoint=raw_spindle.last_samp/1000-6
                    start,stop=raw_spindle.time_as_index([startPoint,endPoint])
                    S,T = raw_spindle[channelID,start:stop]
                    timePoint=np.random.choice(T,1)
                    if (Threshold_test(timePoint,raw_alpha,raw_spindle,raw_muscle,channelID)) and (spindle_overlapping_test(spindles,timePoint,1.5)) and (used_windows_check(timePoint,used_time_windows,1.5)):
                        Segment,_=cut_segments(raw_spindle,timePoint,channelID)
                        nonspindle[fileName][names].append(Segment)
                        used_time_windows.append([timePoint-1.5,timePoint+1.5])
                        print(timePoint,'non spindle')
                        break
                    else:
                        #print('no non spindle left')
                        pass
            except:
                print('error')
        else:
            pass


result={'spindle':spindle,'non spindle':nonspindle,'spindle time':time_spindle,'nonspindle time':time_nonspindle}
import pickle
with open('single subject.p','wb') as handle:
    pickle.dump(result,handle)


fileName=list(result['spindle'].keys())[0]
channelList=['F3','F4','C3','C4','O1','O2']
X = [];Y=[]
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
X_train, X_test, y_train, y_test = train_test_split(normal_X, YY, test_size=0.20)

#print(y_train)

clf =LogisticRegression(penalty='l2',C=.1,tol=10e-9,fit_intercept=True,solver='liblinear',
                                             max_iter=10e7,multi_class='ovr',n_jobs=-1)



clf.fit(X_train,y_train)
print('logistic,1 fold, 80/20 train/test slipt')
print(classification_report(clf.predict(X_test),y_test))
print('within training data set',clf.score(X_train,y_train))

num=int(input("please enter 1 or 2"))
file_to_read,fileName=eegPinelineDesign.pick_sample_file(EDFfiles,n=num)
channelList = ['F3','F4','C3','C4','O1','O2','ROC','LOC']
raw_filter = eegPinelineDesign.load_data(file_to_read,channelList,11, 16)#spindle pass
raw_alpha=eegPinelineDesign.load_data(file_to_read,channelList,8, 12)#alpha pass
raw_spindle=eegPinelineDesign.load_data(file_to_read,channelList,11, 16)#spindle pass
raw_muscle=eegPinelineDesign.load_data(file_to_read,channelList,30, 40)#
channelList = ['F3','F4','C3','C4','O1','O2']
raw_filter.pick_channels(channelList)

time_label={};resolution = 0.02
for names in channelList:
    time_label[names]=[]
    

centerPoint = 0+5;cnt = 0
while raw_filter.last_samp/1000 - centerPoint > 1.5:
    if (cnt % 500) == 0:
        print(centerPoint)
    for ii,names in enumerate(channelList):
    
        if Threshold_test(centerPoint,raw_alpha,raw_spindle,raw_muscle,ii):
            tempSegment,timeSpan=cut_segments(raw_filter,centerPoint,ii,windowsize = 1.5)
            predictedLabel = clf.predict(tempSegment[0,:3000])
            time_label[names].append([centerPoint,predictedLabel])
        else:
            time_label[names].append([centerPoint,0])

    centerPoint += resolution;cnt += 1
print('done')

with open('single subject prediction.p','wb') as handle:
    pickle.dump(time_label,handle)
