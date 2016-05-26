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
eegPinelineDesign.change_file_directory('C:/Users/ning/Downloads/training set')
EDFfiles, Annotationfiles = eegPinelineDesign.split_type_of_files()

#from eegPinelineDesign import CenterAtPeakOfWindow,Threshold_test,spindle_overlapping_test,used_windows_check,cut_segments
def Threshold_test(timePoint,raw_alpha,raw_spindle,raw_muscle,channelID,threshold = 0.5,steps=20):
    result = []
    for windowsize in np.linspace(0.1,5,steps):
        startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
        start,stop=raw_spindle.time_as_index([startPoint,endPoint])
        filter_alpha,timeSpan = raw_alpha[channelID,start:stop]
        RMS_alpha=np.sqrt(sum(filter_alpha[0,:]**2)/len(filter_alpha[0,:]))
        
        filter_spindle,_=raw_spindle[channelID,start:stop]
        RMS_spindle=np.sqrt(sum(filter_spindle[0,:]**2)/len(filter_spindle[0,:]))
        
        filter_muscle,_=raw_muscle[channelID,start:stop]
        RMS_muscle=np.sqrt(sum(filter_muscle[0,:]**2)/len(filter_muscle[0,:]))
    
        if (RMS_alpha/RMS_spindle <1.2) and (RMS_muscle < 5*10e-4):
            result.append(1)
        else:
            result.append(0)
    result = np.array(result)
    return sum(result[result==1])/len(result) >=threshold
    
    
def getOverlap(a,b):
    return max(0,min(a[1],b[1]) - max(a[0],b[0]))
def spindle_overlapping_test(spindles,timePoint,windowsize,tolerance=0.01):
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    return all(getOverlap([startPoint,endPoint],[instance-windowsize,instance+windowsize])<=tolerance for instance in spindles)

def used_windows_check(timePoint,used_time_windows,windowsize,tolerance=0.01):
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    return all(getOverlap([startPoint,endPoint],[lower,upper])<=tolerance for (lower,upper) in used_time_windows)

def cut_segments(raw,center,channelIndex,windowsize = 1.5):
    startPoint=center-windowsize;endPoint=center+windowsize
    start,stop=raw.time_as_index([startPoint,endPoint])
    tempSegment,timeSpan=raw[channelIndex,start:stop]
    return tempSegment,timeSpan
    
    
spindle={};nonspindle={};time_spindle={};time_nonspindle={}
file_to_read,fileName=eegPinelineDesign.pick_sample_file(EDFfiles,n=0)
channelList = ['F3','F4','C3','C4','O1','O2','ROC','LOC']
raw = eegPinelineDesign.load_data(file_to_read,channelList,None, 100)#low pass data
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
used_time_windows=[]
time_spindle[fileName]={};time_nonspindle[fileName]={}
spindle[fileName]={};nonspindle[fileName]={}
for channelID, names in enumerate(channelList):
    time_spindle[fileName][names]=[];time_nonspindle[fileName][names]=[]
    spindle[fileName][names]=[];nonspindle[fileName][names]=[]
    for items in spindles:
        if Threshold_test(items,raw_alpha,raw_spindle,raw_muscle,channelID):
            centered_marker=eegPinelineDesign.CenterAtPeakOfWindow(items,2,raw_spindle,channelID)
            Segment,_ = cut_segments(raw_spindle,centered_marker,channelID)
            spindle[fileName][names].append(Segment)
            time_spindle[fileName][names].append(centered_marker)
            print(fileName,names,centered_marker,'spindle')
            for max_iteration in range(10000):
                startPoint=6;endPoint=raw.last_samp/1000-6
                start,stop=raw.time_as_index([startPoint,endPoint])
                S,T = raw_spindle[channelID,start:stop]
                timePoint=np.random.choice(T,1)
                if (Threshold_test(timePoint,raw_alpha,raw_spindle,raw_muscle,channelID)) and (spindle_overlapping_test(spindles,timePoint,1.5)) and (used_windows_check(timePoint,used_time_windows,1.5)):
                    Segment,_=cut_segments(raw_spindle,timePoint,channelID)
                    nonspindle[fileName][names].append(Segment)
                    used_time_windows.append([timePoint-1.5,timePoint+1.5])
                    time_nonspindle[fileName][names].append(timePoint)
                    print(timePoint,'non spindle')
                    break
                else:
                    pass
        else:
            pass