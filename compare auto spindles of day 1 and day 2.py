# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:07:44 2017

@author: ning
"""

import mne
import pandas as pd
import eegPinelineDesign
import numpy as np
from random import shuffle
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt


annotation_files = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
annotation_files = [f for f in annotation_files if ('annotations' in f)]


windowSize=500;syn_channel=3;low_T = 0.4;high_T = 3.4; 
channelList = ['F3','F4','C3','C4','O1','O2']
spindles =[];even=1
load2, load5 = [],[]
for ii in np.arange(11,32):
    
    annotation_files_to_read = [f for f in annotation_files if ('suj'+str(ii) in f)]
    if len(annotation_files_to_read) > 1:
        temp=[]
        for a in annotation_files_to_read:
            annotation = pd.read_csv(a)
            if ii >= 11:
                day = a.split('_')[2][:4]
                for_name='sub'+str(ii)+day
                old = False
            else:
                day = a.split('_')[1]
                for_name = 'sub'+str(ii)+day[0]+'ay'+day[1]
                old = True
            #print(a,day)
            raw_file = [f for f in eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set') if ('fif' in f) and ('suj'+str(ii) in f) and (day in f)]
            #print(raw_file)
            raw = mne.io.read_raw_fif(raw_file[0],preload=True)
            load = raw_file[0].split('_')[1][1]
            print(load)
            if old:
                pass
            else:
                raw.resample(500, npad="auto")
            raw.pick_channels(channelList)
            raw.filter(11,16)
            time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,channelList,annotation,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=low_T,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=high_T,
                                                                                                    front=300,back=100,sleep_stage=True,)
            if load == '2':
                load2.append([ii,len(time_find)])
            elif load == '5':
                load5.append([ii,len(time_find)])
spindles = np.array([load2, load5])
load2_s = np.array([22,25,18,25.5,23,25.5,27,22.5,24,21.5,24,22,25.5,24.5,28,27 ])
print(np.array(load2)[:,0],np.array(load2)[:,1] / load2_s)
fig, ax = plt.subplots(figsize=(8,8))
ax.hist(load2,label='load 2')
ax.hist(load5,label='load 5')
ax.legend()     
        
##### permutation test
#spindles = np.array(spindles)
ps=[]
for aa in range(500):
    difference = np.mean(spindles[1,:] - spindles[0,:])
    vector_spindles = spindles.flatten()
    diff = []
    for ii in range(5000):
        shuffle(vector_spindles)
        permu_spindle = vector_spindles.reshape(spindles.shape)
        diff.append(np.mean(permu_spindle[1,:] - permu_spindle[0,:]))
    ps.append((100-percentileofscore(diff,difference))/100)
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(ps,bins=50,color='blue',label='p values')
ax.axvline(0.05,color='red',label='significant lavel')