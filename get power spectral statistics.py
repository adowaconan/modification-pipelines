# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:28:12 2016

@author: install
"""

import eegPinelineDesign
import mne
from mne.time_frequency import psd_multitaper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33)))

for idx in [29]: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if (str(idx) in folder_to_look) and ('suj' in folder_to_look)]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    for file_to_read in list_file_to_read:    
        raw=mne.io.read_raw_fif(file_to_read[:-5]+'.fif',preload=True,add_eeg_ref=False)
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        picks = mne.pick_types(raw.info,eeg=True,eog=False)
        fmin=0;fmax=50;epoch_length=30
        epochs = eegPinelineDesign.make_overlap_windows(raw,epoch_length=epoch_length)
        psd_dict=dict()
        for names in channelList:
            psd_dict[names]=[]
        for ii,epch in enumerate(epochs):
            eegPinelineDesign.update_progress(ii,len(epochs))
            psds, freqs = psd_multitaper(raw, low_bias=True, tmin=epch[0], tmax=epch[1],
                             fmin=fmin, fmax=fmax, proj=False, picks=picks,
                             n_jobs=1)
            psds = 10 * np.log10(psds)
            for jj,names in enumerate(channelList):
                psd_dict[names].append(psds[jj,:])
        for names in channelList:
            temp = np.vstack(psd_dict[names])