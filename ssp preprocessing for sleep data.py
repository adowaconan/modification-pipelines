# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:03:19 2017

@author: ning

This script will preprocess continue-recording EEG data (sleep data) using 
source spass projection technique. Here we choose 3 sources, so that we don't
throw away too much signal, but there is a trade-off since throwing/keeping signal
is throwing/keeping noise
"""

import os
#import eegPinelineDesign
import mne
from mne.preprocessing import compute_proj_eog
import pandas as pd
import numpy as np

working_dir = 'D:\\NING - spindle\\training set\\'
saving_dir = 'D:\\NING - spindle\\training set\\'
vhdrs = [f for f in os.listdir(working_dir) if ('vhdr' in f) and ('nap.' in f)]

for raw_file in vhdrs:
    montage = mne.channels.read_montage('standard_1020')# read the channel location
    raw = mne.io.read_raw_brainvision('%s%s'%(working_dir,raw_file),preload=True)#,scale=1e6)
    # map the none-name to the channel names because some subjects we don't have the names
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
    # set some channels as stimuli channel, so we could exclude from the list
    raw.set_channel_types({'Aux1':'stim','STI 014':'stim'})
    # if the channles don't have a name, we map them
    if raw.ch_names[0] != 'Fp1':
        raw.rename_channels(chan_dict)
    # take away the last two, which are the stimuli channels
    chan_list=raw.ch_names[:-2]
    raw.pick_channels(chan_list)
    # set eog channels
    raw.set_channel_types({'LOc':'eog','ROc':'eog'})
    # load the montage to the raw object
    raw.set_montage(montage)
    # we don't filter the eog channels
    picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
    raw.filter(None,100,l_trans_bandwidth=0.01,
               h_trans_bandwidth='auto',filter_length=30,picks=picks,)#n_jobs=4)
    raw.notch_filter(np.arange(60,241,60), picks=picks)
    # re-reference to the average
    raw.set_eeg_reference().apply_proj()
    # compute the eog projects, and here is one parameter we will tune: n_eeg. 
    # too high will throw away too much signal with the noise, but too low, we 
    # will keep too much noise
    projs, events = compute_proj_eog(raw,  n_eeg=3, average=True,reject={'eeg':80,'eog':100},)#n_jobs=4)
    #layout = mne.channels.find_layout(raw.info,ch_type='eeg')
    #mne.viz.plot_projs_topomap(projs,layout)
    raw.info['projs'] += projs[1:]
    raw.apply_proj() # apply projects
    fileName = raw_file[:-5]
    raw.save(saving_dir + fileName+'_raw_ssp.fif',proj=True,overwrite=True)
    del raw
#noise_cov=mne.compute_raw_covariance(raw,picks=picks)


working_dir = 'D:\\NING - spindle\\'
saving_dir = 'D:\\NING - spindle\\training set\\'
sub_folders = [f for f in os.listdir(working_dir) if ('suj' in f)]
vhdrs = []
for suj_folder in sub_folders:
    vhdr_ = [f for f in os.listdir(working_dir+suj_folder) if ('vhdr' in f) and ('nap' in f)]
    for v in vhdr_:
        vhdrs.append(working_dir+suj_folder+'/'+v)
        
for raw_file in vhdrs:
    montage = mne.channels.read_montage('standard_1020')
    raw = mne.io.read_raw_brainvision('%s'%(raw_file),preload=True)#,scale=1e6)
    raw.set_channel_types({'Aux1':'stim','STI 014':'stim'})
    chan_list=raw.ch_names[:-2]
    raw.pick_channels(chan_list)
    raw.set_channel_types({'LOc':'eog','ROc':'eog'})
    raw.set_montage(montage)
    picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)
    raw.filter(None,100,l_trans_bandwidth=0.01,
               h_trans_bandwidth='auto',filter_length=30,picks=picks,)#n_jobs=4)
    raw.notch_filter(np.arange(60,241,60), picks=picks)
    raw.set_eeg_reference().apply_proj()
    projs, events = compute_proj_eog(raw,  n_eeg=3, average=True,reject={'eeg':80,'eog':100},)#n_jobs=4)
    #layout = mne.channels.find_layout(raw.info,ch_type='eeg')
    #mne.viz.plot_projs_topomap(projs,layout)
    raw.info['projs'] += projs[1:]
    raw.apply_proj()
    fileName = raw_file.split('/')[1][:-5]
    raw.save(saving_dir + fileName+'_raw_ssp.fif',proj=True,overwrite=True)
    del raw



#raw.filter(11,16)
#raw.pick_channels(raw.ch_names[:-2])
#annotation = pd.read_csv('D:\\NING - spindle\\training set\\suj5_d2final_annotations.txt')
#channelList = ['F3','F4','C3','C4','O1','O2']
#time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
#                                                                                                                              channelList,
#                                                                                                                              annotation,
#                                                                                                                              moving_window_size=1000,
#                                                                                                                              proba=True)
#gold_standard = eegPinelineDesign.read_annotation(raw,'D:\\NING - spindle\\training set\\suj5_d2final_annotations.txt')
#manual_labels,_ = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
