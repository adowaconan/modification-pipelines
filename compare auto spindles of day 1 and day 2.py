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
spindles =[];even=1;durations=[]
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
                load2.append([ii,len(time_find),len(annotation[annotation['Annotation']=='spindle'])])
            elif load == '5':
                load5.append([ii,len(time_find),len(annotation[annotation['Annotation']=='spindle'])])
            durations.append(Duration)
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
for aa in range(100):
    difference = np.mean(spindles[0,:,2]) - np.mean(spindles[1,:,2])
    shape_1 = len(spindles[0,:,1])
    vector_spindles = np.concatenate([spindles[0,:,2],spindles[1,:,2]])
    diff = []
    for ii in range(500):
        shuffle(vector_spindles)
        permu_spindle = vector_spindles
        new_a = permu_spindle[:shape_1]
        new_b = permu_spindle[shape_1:]
        diff.append(np.mean(new_a) - np.mean(new_b))
    ps.append((100-percentileofscore(diff,difference))/100)
fig, ax = plt.subplots(figsize=(8,5))
ax.hist(ps,bins=50,color='blue',label='p values')
ax.axvline(0.05,color='red',label='significant lavel')

#####################################################################################
def Permutation_test(data1, data2, n1=100,n2=100):
    p_values = []
    for simulation_time in range(n1):
        shuffle_difference =[]
        experiment_difference = np.mean(data1,0) - np.mean(data2,0)
        vector_concat = np.concatenate([data1,data2])
        for shuffle_time in range(n2):
            shuffle(vector_concat)
            new_data1 = vector_concat[:len(data1)]
            new_data2 = vector_concat[len(data1):]
            shuffle_difference.append(np.mean(new_data1) - np.mean(new_data2))
        p_values.append(min(percentileofscore(shuffle_difference,experiment_difference)/100,
                            (100-percentileofscore(shuffle_difference,experiment_difference))/100))
    
    return p_values,np.mean(p_values),np.std(p_values)
w_mean = np.mean(np.array(load2)[:,1] / load2_s)
w_std  = np.std(np.array(load2)[:,1] / load2_s)
d_mean = 1.87363
d_std  = 1.283474
w_sample = np.random.normal(w_mean,w_std,size=500)
d_sample = np.random.normal(d_mean,d_std,size=500)
w_sample[w_sample < 0] = 0
d_sample[d_sample < 0] = 0
a,b,c = Permutation_test(w_sample,d_sample)
