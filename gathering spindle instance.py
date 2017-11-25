# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:17:25 2017

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
    for one_spindle_data in tqdm(epochs_spindle.get_data(),desc='spindle loop'): # for each onset of the spindle marked by an annotator
        if one_spindle_data.shape[-1] > 2000:
            data = one_spindle_data[:,:-1]
        else:
            data = one_spindle_data
#        # replace "outlier" (1%) with the trimmed maximum and minimum, I am not sure if this will change anything
#        Trimmed_data = stats.trimboth(data.flatten(),0.01)
#        idx_max = data > Trimmed_data.max()
#        idx_min = data < Trimmed_data.min()
#        data[idx_max] = Trimmed_data.max()
#        data[idx_min] = Trimmed_data.min()
        temp_data.append(data.T)
        instance += 1 # count one instance
        
        if instance >= 500: # when I hit 500 instances, I will go to save
            random_idx = np.random.choice(np.arange(500),size=4,replace=False,)
            temp_data = np.array(temp_data) # put into numpy array
            np.save(saving_dir+'data%d.npy'%save_,temp_data,) # save
            
            plt.close('all')
            fig,axes  = plt.subplots(figsize=(16,16),nrows=2,ncols=2)
            for kk,ax in zip(random_idx,axes.flatten()):
                evoke_ = mne.EvokedArray(temp_data[kk].T,info,)
                mne.viz.plot_evoked_topomap(evoke_,show=False,axes=ax,times=[0.5])
            fig.savefig(saving_dir+'random_selected%d.png'%save_,dpi=300)
            
            instance = 0 # reset the counter
#            temp_data = np.zeros((500,61,2000),) # reset the storing array
            temp_data = []
            save_ +=1
    del data
            
            
temp_data = np.array(temp_data) # put into numpy array
#temp_data = temp_data[temp_data.sum(1).sum(1) !=0] # take out the rows the rows that is not preallocated as zeros
np.save(saving_dir+'data%d.npy'%save_,temp_data,) # save 
plt.close('all')
random_idx = np.random.choice(np.arange(temp_data.shape[0]),size=4,replace=False,)
fig,axes  = plt.subplots(figsize=(16,16),nrows=2,ncols=2)
for kk,ax in zip(random_idx,axes.flatten()):
    evoke_ = mne.EvokedArray(temp_data[kk].T,info,)
    mne.viz.plot_evoked_topomap(evoke_,show=False,axes=ax,times=[0.5])
fig.savefig(saving_dir+'random_selected%d.png'%save_,dpi=300)


