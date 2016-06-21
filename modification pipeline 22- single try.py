# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 19:06:30 2016

@author: ning
"""

import eegPinelineDesign
import numpy as np
import random
import mne
from mne.time_frequency import psd_multitaper
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import label_binarize,scale,normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize,StandardScaler

try:
    eegPinelineDesign.change_file_directory('C:/Users/ning/Downloads/training set')
except:
    pass
EDFfiles, Annotationfiles = eegPinelineDesign.split_type_of_files()


for num, EDFfileName in enumerate(EDFfiles):
    if EDFfileName == 'suj13_l2nap_day2 edited.edf' or EDFfileName =='suj13_l2nap_day2 edited1.edf':
        pass
    else:
        file_to_read,fileName=eegPinelineDesign.pick_sample_file(EDFfiles,n=num)
        channelList = ['F3','F4','C3','C4','O1','O2','ROC','LOC']
        # band pass the data
        raw_narrow = eegPinelineDesign.load_data(file_to_read,channelList,12, 14)
        raw_alpha=eegPinelineDesign.load_data(file_to_read,channelList,8, 12)#alpha pass
        raw_spindle=eegPinelineDesign.load_data(file_to_read,channelList,11, 16)#spindle pass
        raw_muscle=eegPinelineDesign.load_data(file_to_read,channelList,30, 40)#muscle
        # select channels of interests
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw_narrow.pick_channels(channelList)
        raw_alpha.pick_channels(channelList)
        raw_spindle.pick_channels(channelList)
        raw_muscle.pick_channels(channelList)
        # start to calculate RMS per 200 ms
        SlideWindow = 0.2 #resolution
        TimePoint = SlideWindow/2# initial time
        distance_to_end = raw_narrow.last_samp/1000 - TimePoint
        # preallocation
        RMS={}
        RMS['time']=[]
        for names in channelList:
            RMS[names]=[]
        #loop
        while distance_to_end > SlideWindow:
            RMS['time'].append(TimePoint)
            for ii, names in enumerate(channelList):
                tempSegment, _ = eegPinelineDesign.cut_segments(raw_narrow,TimePoint,ii,SlideWindow)
                RMS_temp = np.sqrt(np.sum(tempSegment[0,:]**2))/len(tempSegment[0,:])
                RMS[names].append(RMS_temp)
            TimePoint += SlideWindow
            distance_to_end = raw_narrow.last_samp/1000 - TimePoint
        

        # Get lowest RMS value at each time point
        # Use the lowest RMS value to represent the possible lower bound 
        # across 6 channels, so that when we apply the threshold, the duration
        # of each spindle could hold for a more liberal range
        RMS_common=[]
        for ii, _ in enumerate(RMS['time']):
            points=[]
            for names in channelList:
                points.append(np.array(RMS[names])[ii])
            RMS_common.append(np.min(points))
        RMS_common=np.array(RMS_common)
        # peak detection algorithm provided in jupyter notebook forum
        mph = RMS_common[100:-100].mean() + RMS_common[100:-100].std()
        ints=eegPinelineDesign.detect_peaks(RMS_common[100:-100],mph=mph,mpd=20,show=False) 
        # first 100 and last 100 s are ignored
        time = np.array(RMS['time'])[100:-100][ints]   
        KK=np.zeros((6,len(time)))     
        for jj,items in enumerate(time):
            for ii,names in enumerate(channelList):
                KK[ii,jj]=eegPinelineDesign.Threshold_test(items,raw_alpha,raw_spindle,raw_muscle,ii,windowsize=2.5)
        # depend on how many channels need to pass, the higher, the more conservative
        Time_found = time[KK.sum(axis=0)>3]
        # save some memory
        raw_alpha.close()
        raw_spindle.close()
        raw_muscle.close()
        raw_narrow.close()
        # get the man marked spindles for comparison
        annotation_to_read = [x for x in Annotationfiles if fileName in x]
        file = pd.read_csv(annotation_to_read[0])
        labelFind = re.compile('spindle',eegPinelineDesign.re.IGNORECASE)
        spindles=[]# take existed annotations
        for row in file.iterrows():
            currentEvent = row[1][-1]
            if labelFind.search(currentEvent):
                spindles.append(row[1][0])# time of marker    
        spindles = np.array(spindles)        
        # Take a look how many of instances detected match to the man marked spindles
        match=[]
        mismatch=[]
        for item in spindles:
            if any(abs(item - Time_found)<2):
                match.append(item)
            else:
                mismatch.append(item)
        fig,ax=plt.subplots()
        ax.bar(np.arange(3),[len(match),len(mismatch),len(spindles)],align="center")
        ax.text(0-0.4,len(match)+3,'match rate is %.2f' % (len(match)/len(spindles)))
        ax.text(0-0.4,len(match)+5,'auto detection found %d' % (len(Time_found)))
        ax.set_title(fileName)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(['match','mismatch','man marked spindles'])
        fig.savefig('C:/Users/ning/Downloads/training set/figures/'+fileName+'.png')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        