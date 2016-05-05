# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:39:38 2016

@author: ning
"""

import numpy as np
import random
import mne
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import json
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA,FastICA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from scipy.fftpack import fft,ifft
import math
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from scipy.signal import spectrogram
from mne.preprocessing.ica import ICA
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize


def change_file_directory(path_directory):
    current_directory=os.chdir(path_directory)
    print(os.listdir(current_directory))

def split_type_of_files():
    EDFFind = re.compile("edf", re.IGNORECASE);EDFfiles=[]
    TXTFind = re.compile("txt",re.IGNORECASE);Annotationfiles=[]
    """This function will go through the current directory and 
    look at all the files in the directory. 
        The reason I have this function is because it create a file 
    space for looping the feature extraction"""
    directoryList = os.listdir(os.getcwd())
    for item in directoryList:
        if EDFFind.search(item):
            EDFfiles.append(item)
        elif TXTFind.search(item):
            Annotationfiles.append(item)
    return EDFfiles,Annotationfiles



def pick_sample_file(EDFfile,n=0):
    file_to_read=EDFfile[n]
    fileName=file_to_read.split('.')[0]
    return file_to_read,fileName
    
    
def load_data(file_to_read,channelList,low_frequency=5,high_frequency=50):
    """ not just the data, but also remove artifact by using mne.ICA"""
    raw = mne.io.read_raw_edf(file_to_read,stim_channel=None,preload=True)
    raw.pick_channels(channelList)
    ica = ICA(n_components=None, n_pca_components=None, max_pca_components=None,max_iter=3000,
          noise_cov=None, random_state=0)
    picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
    ica.fit(raw,picks=picks,decim=3,reject=dict(mag=4e-12, grad=4000e-13))
    ica.detect_artifacts(raw,eog_ch='ROC',eog_criterion=0.5)
    clean_raw = ica.apply(raw,exclude=ica.exclude)
    if low_frequency is not None and high_frequency is not None:
        clean_raw.filter(low_frequency,high_frequency)
    elif low_frequency is not None or high_frequency is not None:
        try: 
            clean_raw.filter(low_frequency,500)
        except:
            clean_raw.filter(0,high_frequency)
    else:
        clean_raw = clean_raw
    
    return clean_raw

def annotation_to_labels(TXTfiles,fileName,label='markon',last_letter=-1):
    annotation_to_read=[x for x in TXTfiles if fileName in x]
    file = pd.read_csv(annotation_to_read[0])
    #file['Duration'] = file['Duration'].fillna(0)
    labelFind = re.compile(label,re.IGNORECASE)
    windowLabel=[]
    for row in file.iterrows():
        currentEvent = row[1][-1]
        if (labelFind.search(currentEvent)):
          
            windowLabel.append(currentEvent[-1])
    for idx,items in enumerate(windowLabel):
        if items == ' ':
            windowLabel[idx] = windowLabel[idx -1]
    return windowLabel
def relabel_to_binary(windowLabel,label=['2','3']):
    YLabel=[]
    for row in windowLabel:
        if row[0] == label[0] or row[0] == label[1]:
            YLabel.append(1)
        else:
            YLabel.append(0)
    return YLabel
unit_step=lambda x:0 if x<0 else 1
def structure_to_data(channelList,YLabel,raw,sample_points=1000):
    data={}
    for channel_names in channelList:
        data[channel_names]=[]
    data['label']=[]
    channel_index = mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
    for sample,labels in zip(range(len(YLabel)),YLabel):
        
        try:
            startPoint=30*sample;endPoint=30*(sample+1)
            start,stop=raw.time_as_index([startPoint,endPoint])
            segment,time=raw[channel_index,start:stop]
            
            for idx, channel_names in enumerate(channelList):
                yf = 20*np.log10(np.abs(np.fft.rfft(segment[idx,:sample_points])))
                data[channel_names].append(yf)
            data['label'].append(labels)
        except:
            print('last window is missing due to error','sample that is passed is',sample)
            #data['label']=scipy.delete(YLabel,sample,0)
            pass
            
    return data
def merge_dicts(dict1,dict2):
    for key, value in dict2.items():
        dict1.setdefault(key,[]).extend(value)
    return dict1

def logistic_func(theta, x):
    return 1./(1+np.exp(x.dot(theta)))
def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc
def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)
def grad_desc(theta_values, X, y, lr=10e-8, converge_change=10e-6):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1;#print(i)
    return theta_values, np.array(cost_iter)
def pred_values(theta, X, hard=True,one_sample=False):
    #normalize
    if not one_sample:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob
def SK_to_data(channelList,markPairs,dataLabels,raw):
    data={}
    for channel_names in channelList:
        data[channel_names]=[]
    data['label']=[]
    channel_index,_=dictionary_for_target_channels(channelList,raw)
    for sample,pairs in enumerate(markPairs):
        #print(idx)
        
        start,stop = raw.time_as_index(pairs)
            
        segment,time=raw[channel_index,start:stop]
        try:    
            for idx,channel_names in enumerate(channelList):
                yf = fft(segment[idx,:]);N=100;#print(channel_names,N)
                data[channel_names].append(np.abs(yf[0:100]))
            data['label'].append(dataLabels[sample])
        except:
            continue
       
    return data
def annotation_file(TXTFiles,sample_number=0):
    annotation_to_read=[x for x in TXTfiles if fileName in x]
    file = pd.read_csv(annotation_to_read[0])
    file['Duration'] = file['Duration'].fillna(0)
    return file

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def center_window_by_max_amplitude(raw,time,channelList,windowsWidth=2.0):
    startPoint=time-windowsWidth;endPoint=time+windowsWidth
    start,stop=raw.time_as_index([startPoint,endPoint])
    tempsegment,timespan=raw[:,start:stop]
    centerxval = timespan[np.argmax(abs(tempsegment[ii,:]))]
    startPoint=centerxval-windowsWidth/2;endPoint=centerxval+windowsWidth/2
    start,stop=raw.time_as_index([startPoint,endPoint])
    segment,_=raw[:,start:stop]
    segment_dictionary={}
    for idx,name in enumerate(channelList):
        yf = fft(segment[idx,:])[:50] 
        segment_dictionary[name]= abs(yf)
    return segment_dictionary

def from_time_markers_to_sample(channelList,raw,windowsWidth=2.0):
    data={}
    for names in channelList:
        data[names]=[]
    for moments in time:
        segments=center_window_by_max_amplitude(raw,moments, channelList,windowsWidth=windowsWidth)
        for names in channelList:
            data[names.append(segments[names])]
    return data
    
def normalized(x):
    normalized_x = (x-np.mean(x))/np.dot((x-np.mean(x)),(x-np.mean(x)))
    return normalized_x

def center_by_windowsize(raw,windowsize,num_channels,start,stop):
    segment = np.empty([num_channels,])
    for ii in range(num_channels):
        tempsegment,timespan = raw[ii,start:stop]
        centerxval = timespan[np.argmax(abs(tempsegment))]
        startPoint = centerxval-windowsize/2;endPoint=centerxval+windowsize/2
        start,stop=raw.time_as_index([startPoint,endPoint])
        segment[ii,:],time=raw[ii,start:stop]
        return segment,time
    
    
def add_channels(inst, data, ch_names, ch_types):
    from mne.io import _BaseRaw, RawArray
    from mne.epochs import _BaseEpochs, EpochsArray
    from mne import create_info
    if 'meg' in ch_types or 'eeg' in ch_types:
        return NotImplementedError('Can only add misc, stim and ieeg channels')
    info = create_info(ch_names=ch_names, sfreq=inst.info['sfreq'],
                       ch_types=ch_types)
    if isinstance(inst, _BaseRaw):
        for key in ('buffer_size_sec', 'filename'):
            info[key] = inst.info[key]
        new_inst = RawArray(data, info=info)#, first_samp=inst._first_samps[0])
    elif isinstance(inst, _BaseEpochs):
        new_inst = EpochsArray(data, info=info)
    else:
        raise ValueError('unknown inst type')
    return inst.add_channels([new_inst], copy=True)