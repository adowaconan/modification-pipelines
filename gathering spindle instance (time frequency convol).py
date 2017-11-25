# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:45:41 2017

@author: ning
"""

import os
import mne
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats
from mne.time_frequency import tfr_multitaper,tfr_morlet
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\Spindle_by_Graphical_Features\\'
annotations = [f for f in os.listdir() if ('annotations.txt' in f)] # get all the possible annotation files
fif_data = [f for f in os.listdir() if ('raw_ssp.fif' in f)] # get all the possible preprocessed data, might be more than or less than annotation files
def spindle(x,KEY='spindle'):# iterating through each row of a data frame and matcing the string with "KEY"
    keyword = re.compile(KEY,re.IGNORECASE) # make the keyword
    return keyword.search(x) != None # return true if find a match
instance = 0 # counter for spindle isntances
#temp_data = np.zeros((500,61,2000),) # preallocate empty array to store the 2 seconds long instances
temp_data = []
save_ = 1 # I calculate in total I will have 3300~ instances of spindles, so I will have to save 4 times if I make 500 instances together
for f in tqdm(annotations,desc='annotation loop'):
    temp_ = re.findall('\d+',f) # get the numbers in the string
    
    sub = temp_[0] # the first one will always be subject number
    day = temp_[-1]# the last one will always be the day
    
    if int(sub) < 11: # change a little bit for matching between annotation and raw EEG files
        day = 'd%s' % day
    else:
        day = 'day%s_' % day
    fif_file = [f for f in fif_data if ('suj%s_'%sub in f.lower()) and (day in f)][0]# the .lower() to make sure the consistence of file name cases
    print(sub,day,f,fif_file) # a checking print
    day = temp_[-1]# the last one will always be the day
    if int(sub) < 11: # change a little bit for matching between annotation and raw EEG files
        day = 'day%s' % day
    else:
        day = 'day%s' % day
    raw_fif = mne.io.read_raw_fif(fif_file,preload=True,) # load data
    raw_fif.pick_types(eeg=True,eog=False) # take away the EOG channels
    raw_fif.pick_channels(raw_fif.ch_names[:32]) # only use the first 32 channels. too much channels might be redundent
    raw_fif.filter(8,20,) # filer between 8 - 20 Hz, leaving most of the parameters default
    sfreq = raw_fif.info['sfreq'] # get sampling frequency
    info = raw_fif.info
    info.normalize_proj()
    raw_fif.info = info
    info = raw_fif.info
    
    df = pd.read_csv(f) # read the annotation table
    idx_row = df['Annotation'].apply(spindle) # find the spindle rows
    df_spindle = df[idx_row] # only take the spindle rows and form a data frame
    df_spindle = df_spindle.drop_duplicates(['Onset'])
    events_={'start':(df_spindle['Onset'].values - 0.5)*sfreq,
             'empty':np.zeros(df_spindle.shape[0],dtype=int),
             'code':np.ones(df_spindle.shape[0],dtype=int)}
    events_=pd.DataFrame(events_,dtype=int)
    events_=events_.drop_duplicates(['start'])
    events_=events_[['start','empty','code']].values.astype(int)
    epochs_spindle = mne.Epochs(raw_fif,events_,tmin=0,tmax=2,event_id=1,baseline=None,detrend=0,preload=True)   
    del raw_fif # save memory
    
    freqs = np.arange(8,21,1)
    n_cycles = freqs / 2.
    time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
    power = tfr_multitaper(epochs_spindle,freqs,n_cycles=n_cycles,time_bandwidth=time_bandwidth,return_itc=False,average=False,)
    power.save(saving_dir + 'sub%s_%s'%(sub,day) + '-tfr.h5',overwrite=True)
    del power
            
        

        
    