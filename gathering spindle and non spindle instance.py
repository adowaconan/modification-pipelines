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
saving_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\eventRelated\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
annotations = [f for f in os.listdir() if ('annotations.txt' in f)] # get all the possible annotation files
fif_data = [f for f in os.listdir() if ('raw_ssp.fif' in f)] # get all the possible preprocessed data, might be more than or less than annotation files
def spindle(x,KEY='spindle'):# iterating through each row of a data frame and matcing the string with "KEY"
    keyword = re.compile(KEY,re.IGNORECASE) # make the keyword
    return keyword.search(x) != None # return true if find a match
def get_events(fif,f,validation_windowsize=2,l_threshold=0.4,h_threshold=3.4):

    raw = mne.io.read_raw_fif(fif_file,preload=True)
    anno = pd.read_csv(f)
    model = Filter_based_and_thresholding()
    channelList = raw.ch_names[:32]
    model.channelList = channelList
    model.get_raw(raw)
    model.get_annotation(anno)
    model.validation_windowsize = validation_windowsize
    model.syn_channels = int(len(channelList)/2)
    model.find_onset_duration(l_threshold,h_threshold)
    model.sleep_stage_check()
    spindle_ = anno['Annotation'].apply(spindle)
    df_spindle = anno[spindle_]
    df_spindle['Onset'] = df_spindle['Onset'] - 0.5
    df_spindle['Duration'] = validation_windowsize
    df_auto = pd.DataFrame()
    df_auto['Onset'] = np.array(model.time_find) - np.array(model.Duration)
    df_auto['Duration'] = model.Duration
    df_auto['Annotation'] = 'spindle'
    # make intervals
    df_spindle['offset']=df_spindle['Onset'] + df_spindle['Duration']
    df_auto['offset'] = df_auto['Onset'] + df_auto['Duration']
    spindle_intervals = df_spindle[['Onset','offset']].values
    auto_intervals = df_auto[['Onset','offset']].values
    
    row_idx = [sum([getOverlap(interval,temp) for temp in spindle_intervals]) != 0 for interval in auto_intervals]
    
    events_1 = pd.DataFrame()
    events_1['Onset'] = auto_intervals[row_idx][:,0]
    events_1['middle'] = 1
    events_1['code'] = 1
    
    events_0 = pd.DataFrame()
    events_0['Onset']= auto_intervals[np.invert(row_idx)][:,0]
    events_0['middle'] = 1
    events_0['code'] = 0
    
    events = pd.concat([events_0,events_1])
    events['Onset'] = np.array(events['Onset'] * raw.info['sfreq'],dtype=int)
    del raw
    return events,model
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
    events,model = get_events(fif_file,f)
    
    # reload data
    raw_fif = mne.io.read_raw_fif(fif_file,preload=True,) # load data
    raw_fif.pick_types(eeg=True,eog=False) # take away the EOG channels
    raw_fif.pick_channels(raw_fif.ch_names[:32]) # only use the first 32 channels. too much channels might be redundent
    raw_fif.filter(8,20,) # filer between 8 - 20 Hz, leaving most of the parameters default
    sfreq = raw_fif.info['sfreq'] # get sampling frequency
    info = raw_fif.info
    info.normalize_proj()
    raw_fif.info = info
    info = raw_fif.info
    # refine the events file
    Onsets = model.time_find
    events['Onset'] = np.array((np.array(Onsets) - 1)*raw_fif.info['sfreq'], dtype=int)
    events=events.drop_duplicates(['Onset'])
    events = events[['Onset','middle','code']].values.astype(int)
    event_id = {'spindle':1,'non_spindle':0}
    try:
        epochs_spindle = mne.Epochs(raw_fif,events,tmin=0,tmax=2,event_id=event_id,baseline=None,detrend=0,preload=True) 
        del raw_fif # save memory
        
        freqs = np.arange(8,21,1)
        n_cycles = freqs / 2.
        time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
        power = tfr_multitaper(epochs_spindle,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
        power.info['event'] = events
        power.save(saving_dir + 'sub%s_%s-eventsRelated'%(sub,day) + '-tfr.h5',overwrite=True)
        del power
    except:
        del raw_fif
        print(sub,day,'No matching events found for spindle (event id 1)')
