# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 21:39:38 2016
@author: ning
"""

import numpy as np
import random
import mne
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
from sklearn.preprocessing import label_binarize,scale
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize,StandardScaler
from mne.time_frequency import psd_multitaper

#from obspy.signal.filter import bandpass

def change_file_directory(path_directory):
    '''Change working directory'''
    current_directory=os.chdir(path_directory)
    #print(os.listdir(current_directory))
    return os.listdir(current_directory)

def split_type_of_files():
    EEGFind = re.compile("vhdr", re.IGNORECASE);EEGfiles=[]
    TXTFind = re.compile("txt",re.IGNORECASE);Annotationfiles=[]
    """This function will go through the current directory and
    look at all the files in the directory.
        The reason I have this function is because it create a file
    space for looping the feature extraction"""
    directoryList = os.listdir(os.getcwd())
    for item in directoryList:
        if EEGFind.search(item):
            EEGfiles.append(item)
        elif TXTFind.search(item):
            Annotationfiles.append(item)
    return EEGfiles,Annotationfiles



def pick_sample_file(EEGfile,n=0):
    """I use it as a way to get names for my dictionary variables"""
    file_to_read=EEGfile[n]
    fileName=file_to_read.split('.')[0]
    return file_to_read,fileName

chan_dict={'Ch56': 'TP8', 'Ch61': 'F6', 'Ch3': 'F3', 'Ch45': 'P1', 'Ch14': 'P3', 
           'Ch41': 'C1', 'Ch1': 'Fp1', 'Ch46': 'P5', 'Ch7': 'FC1', 'Ch37': 'F5', 
           'Ch21': 'TP10', 'Ch8': 'C3', 'Ch11': 'CP5', 'Ch28': 'FC6', 'Ch17': 'Oz', 
           'Ch39': 'FC3', 'Ch38': 'FT7', 'Ch58': 'C2', 'Ch33': 'AF7', 'Ch48': 'PO3', 
           'Ch9': 'T7', 'Ch49': 'POz', 'Ch2': 'Fz', 'Ch15': 'P7', 'Ch20': 'P8', 
           'Ch60': 'FT8', 'Ch57': 'C6', 'Ch32': 'Fp2', 'Ch29': 'FC2', 'Ch59': 'FC4', 
           'Ch35': 'AFz', 'Ch44': 'CP3', 'Ch47': 'PO7', 'Ch30': 'F4', 'Ch62': 'F2', 
           'Ch4': 'F7', 'Ch24': 'Cz', 'Ch31': 'F8', 'Ch64': 'ROc', 'Ch23': 'CP2', 
           'Ch25': 'C4', 'Ch40': 'FCz', 'Ch53': 'P2', 'Ch19': 'P4', 'Ch27': 'FT10', 
           'Ch50': 'PO4', 'Ch18': 'O2', 'Ch55': 'CP4', 'Ch6': 'FC5', 'Ch12': 'CP1', 
           'Ch16': 'O1', 'Ch52': 'P6', 'Ch5': 'FT9', 'Ch42': 'C5', 'Ch36': 'F1', 
           'Ch26': 'T8', 'Ch51': 'PO8', 'Ch34': 'AF3', 'Ch22': 'CP6', 'Ch54': 'CPz', 
           'Ch13': 'Pz', 'Ch63': 'LOc', 'Ch43': 'TP7'}
def load_data(file_to_read,low_frequency=.1,high_frequency=50,eegReject=80,
              eogReject=180,n_ch=-2):
    """ not just load the data, but also remove artifact by using mne.ICA
        Make sure 'LOC' or 'ROC' channels are in the channel list, because they
        are used to detect muscle and eye blink movements"""
        
        
    """
    file_to_read: file name with 'vhdr'
    low_frequency: high pass cutoff point
    high_frequency: low pass cutoff point
    eegReject: change in amplitude in microvoltage of eeg channels
    eogReject: change in amplitude in microvoltage of eog channels
    n_ch: index of channel list, use '-2' for excluding AUX and stimuli channels
    """
    c=200
    raw = mne.io.read_raw_brainvision(file_to_read,scale=1e6,preload=True)
    if 'LOc' in raw.ch_names:
        try:
            #raw.resample(500,npad='auto')
            #chan_list=['F3','F4','C3','C4','O1','O2','ROc','LOc']
            raw.set_channel_types({'LOc':'eog','ROc':'eog'})
            chan_list=raw.ch_names[:n_ch]#exclude AUX and stim channel
            if 'LOc' not in chan_list:
                chan_list.append('LOc')
            if 'ROc' not in chan_list:
                chan_list.append('ROc')
        
            raw.pick_channels(chan_list)
            picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
            raw.filter(None,c,l_trans_bandwidth=0.01,
                       h_trans_bandwidth='auto',filter_length=30,picks=picks)
            noise_cov=mne.compute_raw_covariance(raw.set_eeg_reference(),picks=picks)# re-referencing to average
            raw.notch_filter(np.arange(60,241,60), picks=picks)
            reject = dict(eeg=eegReject,eog=eogReject)
        
            ica = mne.preprocessing.ICA(n_components=0.95,n_pca_components =.95,
                                        max_iter=3000,method='extended-infomax',
                                        noise_cov=noise_cov, random_state=0)
            projs, raw.info['projs'] = raw.info['projs'], []
            ica.fit(raw,picks=picks,start=0,stop=raw.last_samp,decim=2,reject=reject,tstep=2.)
            raw.info['projs'] = projs
            ica.detect_artifacts(raw,eog_ch=['LOc','ROc'],
                                 eog_criterion=0.4,skew_criterion=2,kurt_criterion=2,var_criterion=2)
            try:                     
                a,b=ica.find_bads_eog(raw)
                ica.exclude += a
            except:
                pass
        except:
        
            print('alternative')
            pass
            raw = mne.io.read_raw_brainvision(file_to_read,scale=1e6,preload=True)
            raw.resample(500)
            #chan_list=['F3','F4','C3','C4','O1','O2','ROc','LOc']            
            chan_list=raw.ch_names[:n_ch]
            if 'LOc' not in chan_list:
                chan_list.append('LOc')
            if 'ROc' not in chan_list:
                chan_list.append('ROc')
    
            raw.pick_channels(chan_list)
    
            raw.set_channel_types({'LOc':'eog','ROc':'eog'})
            picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=True,stim=False)
            noise_cov=mne.compute_raw_covariance(raw.add_eeg_average_proj(),picks=picks)
            raw.notch_filter(np.arange(60,241,60), picks=picks)
            reject = dict(eeg=eegReject,
                      eog=eogReject)
            raw.filter(1,c)
            raw_proj = mne.compute_proj_raw(raw,n_eeg=1,reject=reject)
            eog_proj,ev = mne.preprocessing.compute_proj_eog(raw,n_eeg=1,average=True,reject=reject,
                                                 l_freq=1,h_freq=c,
                                                 eog_l_freq=1,eog_h_freq=c)
    
            try:
                raw.info['projs'] += eog_proj
            except:
                pass
            raw.info['projs'] += raw_proj
            raw.apply_proj()
            ica = mne.preprocessing.ICA(n_components=0.95,n_pca_components =.95,
                                        max_iter=3000,method='extended-infomax',
                                        noise_cov=noise_cov, random_state=0)
            ica.fit(raw,start=0,stop=raw.last_samp,decim=3,reject=reject,tstep=2.)
            ica.detect_artifacts(raw,eog_ch=['LOc', 'ROc'],eog_criterion=0.4,
                                 skew_criterion=1,kurt_criterion=1,var_criterion=1)
            
            a,b=ica.find_bads_eog(raw)
            ica.exclude += a
    else:
        print('no channel names')
        
        raw = mne.io.read_raw_brainvision(file_to_read,scale=1e4,preload=True)
        #raw.resample(500)
        raw.rename_channels(chan_dict)
        chan_list=raw.ch_names[:n_ch]
        if 'LOc' not in chan_list:
            chan_list.append('LOc')
        if 'ROc' not in chan_list:
            chan_list.append('ROc')

        raw.pick_channels(chan_list)
        picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
        raw.filter(None,c,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
        noise_cov=mne.compute_raw_covariance(raw.set_eeg_reference(),picks=picks)# re-referencing to average
        raw.notch_filter(np.arange(60,241,60), picks=picks)
        reject = dict(eeg=eegReject,eog=eogReject)
    
        ica = mne.preprocessing.ICA(n_components=0.9,n_pca_components =.9,
                                    max_iter=30,method='extended-infomax',
                                    noise_cov=noise_cov, random_state=0)
        ica.fit(raw,picks=picks,start=0,decim=2,reject=reject,tstep=2.)
        ica.detect_artifacts(raw,eog_ch=['LOc','ROc'],
                             eog_criterion=0.4,skew_criterion=2,kurt_criterion=2,var_criterion=2)
        try:                     
            a,b=ica.find_bads_eog(raw)
            ica.exclude += a
        except:
            pass


    clean_raw = ica.apply(raw,exclude=ica.exclude)
    if low_frequency is not None and high_frequency is not None:
        clean_raw.filter(low_frequency,high_frequency,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
    elif low_frequency is not None or high_frequency is not None:
        try:
            clean_raw.filter(low_frequency,200,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
        except:
            clean_raw.filter(1,high_frequency,l_trans_bandwidth=0.01,
                   h_trans_bandwidth='auto',filter_length=30,picks=picks)
    else:
        clean_raw = clean_raw
    #clean_raw.plot_psd(fmax=50)
    return clean_raw

def annotation_to_labels(TXTfiles,fileName,label='markon',last_letter=-1):
    """This only works on very particular data structure file."""
    annotation_to_read=[x for x in TXTfiles if fileName in x]
    file = pd.read_csv(annotation_to_read[0])
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
    """Become useless after several changes"""
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
    '''The function goes through all channels and return data.frame of
       centered data'''
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


def CenterAtPeakOfWindow(timePoint,windowSize,raw,channelIndex):
    '''Simplification of the function above, return only the centered data time
       point.'''
    filter_tempSegment,timeSpan = cut_segments(raw,timePoint,channelIndex)
    peakInd = np.array(find_peaks_cwt(filter_tempSegment[0,:],np.arange(1,500)))
    max_in_peakInd=np.argmax(abs(filter_tempSegment[0,peakInd]))
    centerxval=timeSpan[peakInd[max_in_peakInd]]
    return centerxval

def from_time_markers_to_sample(channelList,raw,windowsWidth=2.0):
    data={}
    for names in channelList:
        data[names]=[]
    for moments in time:
        segments=center_window_by_max_amplitude(raw,moments, channelList,windowsWidth=windowsWidth)
        for names in channelList:
            data[names.append(segments[names])]
    return data

def standardized(x):
    normalized_x = (x-np.mean(x))/np.std(x)
    return normalized_x



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

def cut_segments(raw,center,channelIndex,windowsize = 1.5):
    startPoint=center-windowsize;endPoint=center+windowsize
    start,stop=raw.time_as_index([startPoint,endPoint])
    tempSegment,timeSpan=raw[channelIndex,start:stop]
    return tempSegment,timeSpan


def Threshold_test(timePoint,raw,channelID,windowsize=2.5):
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    start,stop=raw.time_as_index([startPoint,endPoint])
    se,timeSpan=raw[channelID,start:stop]

    filter_alpha=mne.filter.band_pass_filter(se,1000,8,12)
    filter_spindle=mne.filter.band_pass_filter(se,1000,11,16)
    filter_muscle=mne.filter.band_pass_filter(se,1000,30,40)

    RMS_alpha=np.sqrt(sum(filter_alpha[0,:]**2)/len(filter_alpha[0,:]))
    RMS_spindle=np.sqrt(sum(filter_spindle[0,:]**2)/len(filter_spindle[0,:]))
    RMS_muscle=np.sqrt(sum(filter_muscle[0,:]**2)/len(filter_muscle[0,:]))

    if (RMS_alpha/RMS_spindle <1.2) or (RMS_muscle < 5*10e-4):
        return True
    else:
        return False


def getOverlap(a,b):
    return max(0,min(a[1],b[1]) - max(a[0],b[0]))
def intervalCheck(a,b):#a is an array and b is a point
    return a[0] <= b <= a[1]
def spindle_overlapping_test(spindles,timePoint,windowsize,tolerance=0.01):
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    return all(getOverlap([startPoint,endPoint],[instance-windowsize,instance+windowsize])<=tolerance for instance in spindles)

def used_windows_check(timePoint,used_time_windows,windowsize,tolerance=0.01):
    startPoint=timePoint-windowsize;endPoint=timePoint+windowsize
    return all(getOverlap([startPoint,endPoint],[lower,upper])<=tolerance for (lower,upper) in used_time_windows)

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)
    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)
    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)
    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])


    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def window_rms(a, window_size):
  a2 = np.power(a,2)
  window = scipy.signal.gaussian(window_size,(window_size/.68)/2)
  return np.sqrt(np.convolve(a2, window, 'same')/len(a2)) * 1e2


def distance_check(list_of_comparison, time):
    list_of_comparison=np.array(list_of_comparison)
    condition = list_of_comparison - time < 1
    return condition


def RMS_pass(pass_,time,RMS):
    temp = []
    up = np.where(np.diff(pass_.astype(int))>0)
    down = np.where(np.diff(pass_.astype(int))<0)
    if (up[0].shape > down[0].shape) or (up[0].shape < down[0].shape):
        size = np.min([up[0].shape,down[0].shape])
        up = up[0][:size]
        down = down[0][:size]
    C = np.vstack((up,down))

    for pairs in C.T:
        if 0.5 < (time[pairs[1]] - time[pairs[0]]) < 2:
            TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
            SegmentForPeakSearching = RMS[pairs[0]:pairs[1]]
            temp_temp_time = time[pairs[0]:pairs[1]]
            ints_temp = np.argmax(SegmentForPeakSearching)
            temp.append(temp_temp_time[ints_temp])

    return temp

def RMS_calculation(intervals,dataSegment,mul):
    segment = dataSegment[0,:]
    time = np.linspace(intervals[0],intervals[1],len(segment))
    RMS = window_rms(segment,200)
    mph=scipy.stats.trim_mean(RMS,0.05) + mul * RMS.std()
    pass_=RMS > scipy.stats.trim_mean(RMS,0.05)
    peak_time=RMS_pass(pass_,time,RMS)
    return peak_time,RMS,time


def find_time(peak_time,number=3):
    time_find=[]
    for item in peak_time['mean']:
        temp_timePoint=[]
        channelList = ['F3','F4','C3','C4','O1','O2']
        for ii,names in enumerate(channelList):
            if len(peak_time[names]) == 0:
                pass
            else:
                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(float(x[1])-float(item)))[1])
        try:
            if np.sum((abs(np.array(temp_timePoint) - item)<1).astype(int))>number:
                time_find.append(item)
        except:
            pass

    return time_find

def validation(val_file,result,tol=1):
    file2 = pd.read_csv(val_file,sep=',')
    labelFind = re.compile('spindle',re.IGNORECASE)
    spindles=[]# take existed annotations
    for row in file2.iterrows():
        currentEvent = row[1][-1]
        if labelFind.search(currentEvent):
            spindles.append(row[1][0])# time of marker
    spindles = np.array(spindles)

    peak_time = result['Onset'].values
    Time_found = peak_time
    match=[]
    mismatch=[]
    for item in Time_found:
        if any(abs(item - spindles)<tol):
            match.append(item)
        else:
            mismatch.append(item)
    return spindles, match, mismatch
from scipy.stats import hmean,trim_mean
def EEGpipeline_by_epoch(file_to_read,validation_file,lowCut=10,highCut=18,majority=3,mul=0.8):
    raw = mne.io.read_raw_fif(file_to_read,preload=True,add_eeg_ref=False)
    
    raw.filter(lowCut,highCut,l_trans_bandwidth=0.1)
    channelList = ['F3','F4','C3','C4','O1','O2']
    raw.pick_channels(channelList)
    time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at=get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,moving_window_size=200,threshold=mul,syn_channels=majority,l_freq=lowCut,h_freq=highCut,l_bound=0.5,h_bound=2)
    
    print('finish loading data')
    file2 = pd.read_csv(validation_file,sep=',')
    labelFind = re.compile('Marker: Markon: 2',re.IGNORECASE)
    stage2=[]# take existed annotations
    for row in file2.iterrows():
        currentEvent = row[1][-1]
        if labelFind.search(currentEvent):
            stage2.append([row[1][0],row[1][0]+30])# time of marker
    stage2 = np.array(stage2)
    print('finish loading annotations')
    result = pd.DataFrame({'Onset':np.array(time_find),'Duration':Duration})
    result['Annotation']='spindle'
    spindles, match, mismatch=validation(val_file=validation_file,result=result,tol=1)

    return peak_time, result,spindles, match, mismatch
def EEGpipeline_by_total(file_to_read,validation_file,lowCut=10,highCut=18,majority=3,mul=0.8):
    channelList = ['F3','F4','C3','C4','O1','O2']
    raw = load_data(file_to_read,lowCut,highCut,180)
    raw.pick_channels(channelList)
    print('finish loading data')

    time = np.linspace(0,raw._data[0,:].shape[0]/1000,raw._data[0,:-1].shape[0])
    RMS = np.zeros((6,raw._data[0,:].shape[0]))
    peak_time={}
    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        dataSegment,temptime = raw[ii,:raw.last_samp]
        peak_time[names],RMS[ii,:],time=RMS_calculation([temptime[0],temptime[-1]],dataSegment,mul)

    peak_time['mean']=[]
    RMS_mean=hmean(RMS)
    RMS_mean = np.convolve(RMS_mean, 1000, 'same')# to smooth or to down sampling
        #ax1.plot(time,RMS_mean,color='k',alpha=0.3)
    mph = RMS_mean.mean() + mul * RMS_mean.std()
    pass_ = RMS_mean > RMS_mean.mean()
    peak_time['mean']=RMS_pass(pass_,time,RMS_mean)

    result = pd.DataFrame({'Onset':time_find})
    result['Annotation']='spindle'
    result = result[result.Onset > 30]
    result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 60)]
    spindles, match, mismatch=validation(val_file=validation_file,result=result,tol=1)

    return peak_time, result,spindles, match, mismatch

def TS_analysis(raw,epch,picks,l_freq=8,h_freq=12):
    psd_,f=psd_multitaper(raw,tmin=epch[0],tmax=epch[1],fmin=l_freq,fmax=h_freq,picks=picks,n_jobs=-1)
    return psd_,f

def make_overlap_windows(raw,epoch_length=10):
    candidates = np.arange(raw.first_samp/1000, raw.last_samp/1000,epoch_length/2)
    epochs=[]
    for ii,item in enumerate(candidates):
        #print(ii,len(candidates))
        if ii + 2 > len(candidates)-1:
            break
        else:
            epochs.append([item,candidates[ii+2]])
    return np.array(epochs)


def regressionline(intercept,s,x_range):
    try:
        x = np.array(x_range)
        y = intercept*np.ones(len(x)) + s*x
    except:
        y = intercept + s*x
    return y
def epoch_activity(raw,picks,epoch_length=10):
    # make epochs based on epoch length (10 secs), and overlapped by half of the window
    epochs = make_overlap_windows(raw,epoch_length=epoch_length)
    
     # preallocate
    alpha_C=[];DT_C=[];ASI=[];activity=[];ave_activity=[];slow_spindle=[];fast_spindle=[]
    psd_delta1=[];psd_delta2=[];psd_theta=[];psd_alpha=[];psd_beta=[];psd_gamma=[]

    print('calculating power spectral density')
    for ii,epch in enumerate(epochs):
        # monitor the progress (urgly useless)
        update_progress(ii,len(epochs))
        psds,f = TS_analysis(raw,epch,picks,0,40)
        psds = psds[0]
        psds = 10*np.log10(psds)#rescale
        temp_psd_delta1 = psds[np.where((f<=2))]# delta 1 band
        temp_psd_delta2 = psds[np.where(((f>=2) & (f<=4)))]# delta 2 band
        temp_psd_theta  = psds[np.where(((f>=4) & (f<=8)))]# theta band
        temp_psd_alpha  = psds[np.where(((f>=8) & (f<=12)))]# alpha band
        temp_psd_beta   = psds[np.where(((f>=12) & (f<=20)))]# beta band
        temp_psd_gamma  = psds[np.where((f>=20))] # gamma band
        temp_slow_spindle = psds[np.where((f>=10) & (f<=12))]# slow spindle
        temp_fast_spindle = psds[np.where((f>=12) & (f<=14))]# fast spindle

        temp_activity = [temp_psd_delta1.mean(),
                         temp_psd_delta2.mean(),
                         temp_psd_theta.mean(),
                         temp_psd_alpha.mean(),
                         temp_psd_beta.mean(),
                         temp_psd_gamma.mean()]

        temp_ASI = temp_psd_alpha.mean() /( temp_psd_delta2.mean() + temp_psd_theta.mean())

        alpha_C.append(temp_psd_alpha.mean())
        DT_C.append(temp_psd_delta2.mean() + temp_psd_theta.mean())
        ASI.append(temp_ASI)
        ave_activity.append(temp_activity)
        activity.append(psds[:np.where(f<=20)[0][-1]])#zoom in to beta
        slow_spindle.append(temp_slow_spindle)
        fast_spindle.append(temp_fast_spindle)
        psd_delta1.append(temp_psd_delta1);psd_delta2.append(temp_psd_delta2)
        psd_theta.append(temp_psd_theta);psd_alpha.append(temp_psd_alpha)
        psd_beta.append(temp_psd_beta);psd_gamma.append(temp_psd_gamma)
    slow_range=f[np.where((f>=10) & (f<=12))];fast_range=f[np.where((f>=12) & (f<=14))]
    return alpha_C,DT_C,ASI,activity,ave_activity,psd_delta1,psd_delta2,psd_theta,psd_alpha,psd_beta,psd_gamma,slow_spindle,fast_spindle,slow_range,fast_range,epochs

def mean_without_outlier(data):
    outlier_threshold = data.mean() + data.std()*3
    temp_data = data[np.logical_and(-outlier_threshold < data, data < outlier_threshold)]
    return temp_data.mean()
def trimmed_std(data,percentile):
    temp=data.copy()
    temp.sort()
    percentile = percentile / 2
    low = int(percentile * len(temp))
    high = int((1. - percentile) * len(temp))
    return temp[low:high].std(ddof=0)
def get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,moving_window_size=200,threshold=.9,syn_channels=3,l_freq=0,h_freq=200,l_bound=0.5,h_bound=2,tol=1):
    mul=threshold;nn=3.5
    
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    peak_time={} #preallocate
    fig=plt.figure(figsize=(40,40))
    ax=plt.subplot(311)
    ax1=plt.subplot(312,sharex=ax)
    ax2=plt.subplot(313,sharex=ax)
    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        segment,_ = raw[ii,:]
        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of 200ms
        mph = trim_mean(RMS[ii,100000:-30000],0.05) + mul * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
        mpl = trim_mean(RMS[ii,100000:-30000],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
        pass_ = RMS[ii,:] > mph

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
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl:
                    temp_temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time[names].append(temp_temp_time[ints_temp])
                    ax.scatter(temp_temp_time[ints_temp],mph+0.1*mph,marker='s',
                               color='blue')
        ax.plot(time,RMS[ii,:],alpha=0.2,label=names)
        ax2.plot(time,segment[0,:],label=names,alpha=0.3)
        ax2.set(xlabel="time",ylabel="$\mu$V",xlim=(time[0],time[-1]),title=file_to_read[:-5]+' band pass %.1f - %.1f Hz' %(l_freq,h_freq))
        ax.set(xlabel="time",ylabel='RMS Amplitude',xlim=(time[0],time[-1]),title='auto detection on each channels')
        ax1.set(xlabel='time',ylabel='Amplitude')
        ax.axhline(mph,color='r',alpha=0.03)
        ax2.legend();ax.legend()

    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    ax1.plot(time,RMS_mean,color='k',alpha=0.3)
    mph = trim_mean(RMS_mean[100000:-30000],0.05) + mul * RMS_mean.std()
    mpl = trim_mean(RMS_mean[100000:-30000],0.05) + nn * RMS_mean.std()
    pass_ =RMS_mean > mph
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
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
    for pairs in C.T:
        
        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
            TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
            if np.max(SegmentForPeakSearching)< mpl:
                temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time['mean'].append(temp_time[ints_temp])
                peak_at.append(SegmentForPeakSearching[ints_temp])
                ax1.scatter(temp_time[ints_temp],mph+0.1*mph,marker='s',color='blue')
                duration_temp = time[pairs[1]] - time[pairs[0]]
                duration.append(duration_temp)
    ax1.axhline(mph,color='r',alpha=1.)
    ax1.set_xlim([time[0],time[-1]])


    time_find=[];mean_peak_power=[];Duration=[]
    for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
        temp_timePoint=[]
        for ii, names in enumerate(channelList):
            try:
                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
            except:
                temp_timePoint.append(item + 2)
        try:
            if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
                time_find.append(float(item))
                mean_peak_power.append(PEAK)
                Duration.append(duration_time)
        except:
            pass
    return time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at
def recode_annotation(x):
    if re.compile(': w',re.IGNORECASE).search(x):
        return 0
    elif re.compile(':w',re.IGNORECASE).search(x):
        return 0
    elif re.compile('1',re.IGNORECASE).search(x):
        return 1
    elif re.compile('2',re.IGNORECASE).search(x):
        return 2
    elif re.compile('SWS',re.IGNORECASE).search(x):
        return 3
    elif re.compile('3',re.IGNORECASE).search(x):
        return 3
    else:
        print('error')
        pass


def dist(x1,y1, x2,y2, x3,y3): # x3,y3 is the point
    px = x2-x1
    py = y2-y1

    something = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = math.sqrt(dx*dx + dy*dy)

    return dist
def pass_function(distance):
    pass_ = distance > distance.mean()
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
    
    return C
def pass_(distance):
    
    C = pass_function(distance)
    
    up = C[0,:];down=C[1,:]
    tempD= np.inf
    for u,d in zip(up,down):
        if u - np.argmax(distance)<tempD:
            tempD = np.abs(u - np.argmax(distance))
            up_ = u
            down_=d
    return (up_,down_)
from scipy.stats import linregress
def find_title_peak(x,y):
    x=x[:100];y=y[:100]
    s,intercept,_,_,_ = linregress(x=x,y=y)
    y_pre = regressionline(intercept,s,x)

    A = y - y_pre
    A = A[10:]
    try:
        C = pass_(A)
    except:
        C = (np.argmax(A)-20,np.argmax(A)+20)
    idx_freq_range=C
    idx_devia = np.max([np.abs(np.argmax(A)-idx_freq_range[0]+10),np.abs(np.argmax(A)-idx_freq_range[1]+10)])
    maxArg = np.argmax(A)+10
    #print(maxArg,idx_devia)
    return idx_devia,maxArg
    
def spindle_validation_step1(raw,channelList,file_to_read,moving_window_size=200,threshold=.9,syn_channels=3,l_freq=0,h_freq=200,l_bound=0.5,h_bound=2,tol=1):
    nn=3.5
    
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    peak_time={} #preallocate

    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        segment,_ = raw[ii,:]
        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of 200ms
        mph = trim_mean(RMS[ii,100000:-30000],0.05) + threshold * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
        mpl = trim_mean(RMS[ii,100000:-30000],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
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
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl:
                    temp_temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time[names].append(temp_temp_time[ints_temp])
                    
        

    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    
    mph = trim_mean(RMS_mean[100000:-30000],0.05) + threshold * RMS_mean.std()
    mpl = trim_mean(RMS_mean[100000:-30000],0.05) + nn * RMS_mean.std()
    pass_ =RMS_mean > mph
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
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
    for pairs in C.T:
        
        if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
            TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
            if np.max(SegmentForPeakSearching)< mpl:
                temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time['mean'].append(temp_time[ints_temp])
                peak_at.append(SegmentForPeakSearching[ints_temp])
                duration_temp = time[pairs[1]] - time[pairs[0]]
                duration.append(duration_temp)
    


    time_find=[];mean_peak_power=[];Duration=[]
    for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
        temp_timePoint=[]
        for ii, names in enumerate(channelList):
            try:
                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
            except:
                temp_timePoint.append(item + 2)
        try:
            if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>=syn_channels:
                time_find.append(float(item))
                mean_peak_power.append(PEAK)
                Duration.append(duration_time)
        except:
            pass
    return time_find,mean_peak_power,Duration,peak_time,peak_at
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    if spindle_duration_fix:
        spindle_start = spindle
        spindle_end   = spindle + 2
        a =  np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
    else:
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        a = np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
def discritized_onset_label_manual(raw,df,spindle_segment):
    discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            #print(time_interval,spindle,spindle_segment)
            if spindle_comparison(time_interval,spindle,spindle_segment):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels
def discritized_onset_label_auto(raw,df,spindle_segment):
    spindle_duration = df['Duration'].values
    discritized_continuous_time = np.arange(100,raw.last_samp/raw.info['sfreq']-100,step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for kk,spindle in enumerate(df['Onset']):
            if spindle_comparison(time_interval,spindle,spindle_duration[kk],spindle_duration_fix=False):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels

def read_annotation(raw, annotation_file):
    #annotation_file = [files for files in file_in_fold if('txt' in files) and (file_to_read.split('_')[0] in files) and (file_to_read.split('_')[1] in files)]
    manual_spindle = pd.read_csv(annotation_file[0])
    manual_spindle = manual_spindle[manual_spindle.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
    manual_spindle = manual_spindle[manual_spindle.Onset > 100] 
    keyword = re.compile('spindle',re.IGNORECASE)
    gold_standard = {'Onset':[],'Annotation':[]}
    for ii,row in manual_spindle.iterrows():
        if keyword.search(row[-1]):
            gold_standard['Onset'].append(float(row.Onset))
            gold_standard['Annotation'].append(row.Annotation)
    gold_standard = pd.DataFrame(gold_standard) 
    return gold_standard 
def stage_check(x):
    import re
    if re.compile('2',re.IGNORECASE).search(x):
        return True
    else:
        return False
def sample_data(time_interval_1,time_interval_2,raw,raw_data,stage_on_off,key='miss',old=False):

    if old:
        if key == 'miss':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,1
        elif key == 'hit':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,1
        elif key == 'fa':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,0
        elif key == 'cr':
            
                
            idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
            temp_data = raw_data[:,idx_start:idx_stop].flatten()
            if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                           tmax=time_interval_2,proj=False,)
                psds = 10* np.log10(psds)

                temp_data = np.concatenate((temp_data,
                                            psds.max(1),
                                            freqs[np.argmax(psds,1)]))
                return temp_data,0
    
    else:
        if key == 'miss':
            
            if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,1
        elif key == 'hit':
            
            if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,1
        elif key == 'fa':
            
            if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,0
        elif key == 'cr':
            
            if (sum(eegPinelineDesign.intervalCheck(k,time_interval_1) for k in stage_on_off) >=1 ) and (sum(eegPinelineDesign.intervalCheck(k,time_interval_2) for k in stage_on_off) >=1):
                idx_start,idx_stop= raw.time_as_index([time_interval_1,time_interval_2]) 
                temp_data = raw_data[:,idx_start:idx_stop].flatten()
                if temp_data.shape[0] == 6*3*raw.info['sfreq']:
                    psds,freqs = psd_multitaper(raw,low_bias=True,tmin=time_interval_1,
                                               tmax=time_interval_2,proj=False,)
                    psds = 10* np.log10(psds)

                    temp_data = np.concatenate((temp_data,
                                                psds.max(1),
                                                freqs[np.argmax(psds,1)]))
                    return temp_data,0