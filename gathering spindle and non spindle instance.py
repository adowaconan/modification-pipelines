# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:07:15 2017

@author: ning
"""
import os
os.chdir('D:/Ning - spindle/')
from eegPinelineDesign import getOverlap#thresholding_filterbased_spindle_searching
from Filter_based_and_thresholding import Filter_based_and_thresholding
import mne
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
from mne.time_frequency import tfr_multitaper,tfr_morlet
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated_11_26_2017\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
annotations = [f for f in os.listdir() if ('annotations.txt' in f)] # get all the possible annotation files
fif_data = [f for f in os.listdir() if ('raw_ssp.fif' in f)] # get all the possible preprocessed data, might be more than or less than annotation files
def spindle(x,KEY='spindle'):# iterating through each row of a data frame and matcing the string with "KEY"
    keyword = re.compile(KEY,re.IGNORECASE) # make the keyword
    return keyword.search(x) != None # return true if find a match
def get_events(fif,f,validation_windowsize=2,l_threshold=0.4,h_threshold=3.4,channelList = None):

    raw = mne.io.read_raw_fif(fif,preload=True)
    anno = pd.read_csv(f)
    model = Filter_based_and_thresholding(l_freq=8,h_freq=20)
    
    if channelList == None:
        channelList = ['F3','F4','C3','C4','O1','O2']
    else:
        channelList = raw.ch_names[:32]
    model.channelList = channelList
    model.get_raw(raw,)
    model.get_epochs(resample=256)
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
    freqs = np.arange(8,21,1)
    n_cycles = freqs / 2.
    time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
    power = tfr_multitaper(epochs_,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
    power.info['event'] = events
    del raw
    return power,epochs_,model
for f in tqdm(annotations,desc='annotation loop'):
    temp_ = re.findall('\d+',f) # get the numbers in the string
    
    sub = temp_[0] # the first one will always be subject number
    day = temp_[-1]# the last one will always be the day
    
    if int(sub) < 11: # change a little bit for matching between annotation and raw EEG files
        day = 'd%s' % day
    else:
        day = 'day%s' % day
    fif_file = [f for f in fif_data if ('suj%s_'%sub in f.lower()) and (day in f)][0]# the .lower() to make sure the consistence of file name cases
    print(sub,day,f,fif_file) # a checking print 
    try:
        power,epochs,model = get_events(fif_file,f,channelList=32)
        epochs.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day)+ '-epo.fif',)
        power.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day) + '-tfr.h5',overwrite=True)
        del power
    except:
        pass
    
#    try:
#        epochs_spindle = mne.Epochs(raw_fif,events,tmin=0,tmax=2,event_id=event_id,baseline=None,detrend=0,preload=True) 
#        del raw_fif # save memory
#        
#        freqs = np.arange(8,21,1)
#        n_cycles = freqs / 2.
#        time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
#        power = tfr_multitaper(epochs_spindle,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
#        power.info['event'] = events
#        power.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day) + '-tfr.h5',overwrite=True)
#        del power
#    except:
#        del raw_fif
        print(sub,day,'No matching events found for spindle (event id 1)')
