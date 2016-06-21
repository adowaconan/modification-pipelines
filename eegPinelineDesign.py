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
#from obspy.signal.filter import bandpass

def change_file_directory(path_directory):
    '''Change working directory'''
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
    """I use it as a way to get names for my dictionary variables"""
    file_to_read=EDFfile[n]
    fileName=file_to_read.split('.')[0]
    return file_to_read,fileName


def load_data(file_to_read,channelList,low_frequency=5,high_frequency=50):
    """ not just load the data, but also remove artifact by using mne.ICA
        Make sure 'LOC' or 'ROC' channels are in the channel list, because they
        are used to detect muscle and eye blink movements"""
    raw = mne.io.read_raw_edf(file_to_read,stim_channel=None,preload=True)
    raw.pick_channels(channelList)
    ica = ICA(n_components=None, n_pca_components=None, max_pca_components=None,max_iter=3000,
          noise_cov=None, random_state=0)
    picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
    ica.fit(raw,picks=picks,decim=3,reject=dict(mag=4e-12, grad=4000e-13))
    ica.detect_artifacts(raw,eog_ch=['ROC','LOC'],eog_criterion=0.5)
    clean_raw = ica.apply(raw,exclude=ica.exclude)
    if low_frequency is not None and high_frequency is not None:
        clean_raw.filter(low_frequency,high_frequency)
    elif low_frequency is not None or high_frequency is not None:
        try:
            clean_raw.filter(low_frequency,500)
        except:
            clean_raw.filter(1,high_frequency)
    else:
        clean_raw = clean_raw
    clean_raw.notch_filter(np.arange(60,241,60), picks=picks)
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

def cut_segments(raw,center,channelIndex,windowsize = 1.5):
    startPoint=center-windowsize;endPoint=center+windowsize
    start,stop=raw.time_as_index([startPoint,endPoint])
    tempSegment,timeSpan=raw[channelIndex,start:stop]
    return tempSegment,timeSpan


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

def Threshold_test(timePoint,raw_alpha,raw_spindle,raw_muscle,channelID,windowsize=2.5):
    
    startPoint=timePoint - windowsize;endPoint=timePoint + windowsize
    start,stop=raw_spindle.time_as_index([startPoint,endPoint])
    filter_alpha,timeSpan = raw_alpha[channelID,start:stop]
    RMS_alpha=np.sqrt(sum(filter_alpha[0,:]**2)/len(filter_alpha[0,:]))

    filter_spindle,_=raw_spindle[channelID,start:stop]
    RMS_spindle=np.sqrt(sum(filter_spindle[0,:]**2)/len(filter_spindle[0,:]))

    filter_muscle,_=raw_muscle[channelID,start:stop]
    RMS_muscle=np.sqrt(sum(filter_muscle[0,:]**2)/len(filter_muscle[0,:]))

    if (RMS_alpha/RMS_spindle <1.2) and (RMS_muscle < 5*10e-4):
        return True
    else:
        return False


def getOverlap(a,b):
    return max(0,min(a[1],b[1]) - max(a[0],b[0]))
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