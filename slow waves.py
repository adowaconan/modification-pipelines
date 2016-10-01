# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:13:47 2016

@author: install
"""

import eegPinelineDesign
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
import pickle
import mne
import pandas as pd
import scipy.stats as sts
#import scipy.signal as signal

def read_pickle(fileName):
    result_name = fileName[:-4] +'.p'
    pkl_file=open(result_name,'rb')
    result=pickle.load(pkl_file)
    pkl_file.close()
    return result
def rescale(M):
    M = np.array(M)
    return (M - M.min())/(M.max() - M.min()) + 1
def PSDW(a,b,c):
    return ((a+b)/c)
folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33)))
Above_zero=list()
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if (str(idx) in folder_to_look) and ('suj' in folder_to_look)]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    for file_to_read in list_file_to_read:  
        #raw = eegPinelineDesign.load_data(file_to_read,low_frequency=None,high_frequency=50,n_ch=-2,eegReject=280)
        raw = mne.io.read_raw_brainvision(file_to_read,scale=1e6,preload=True)
        spindles=pd.read_csv(file_to_read[:-5]+'_fast_spindle.csv')
        epoch_length=10 #10sec
        channelList=['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        raw.filter(None,50)
        epochs=eegPinelineDesign.make_overlap_windows(raw)
        temp_slow=np.zeros((6,len(epochs)))
        for ii,name in enumerate(channelList):
            picks=ii
            s,t=raw[ii,:]
            slow_power=list()
            for jj,epch in enumerate(epochs):
                eegPinelineDesign.update_progress(jj,len(epochs))
                slow,_=eegPinelineDesign.TS_analysis(raw,epch,picks,l_freq=0.1,h_freq=1)
                slow=slow[0]
                slow=10*np.log10(slow)
                slow_power.append(slow.mean())
            temp_slow[ii,:]=np.array(slow_power)
            
        slow_power = temp_slow.mean(0)
        
        pass_=slow_power > slow_power.mean()
        
        up = np.where(np.diff(pass_.astype(int))>0)
        down = np.where(np.diff(pass_.astype(int))<0)
        up = up[0]
        down = down[0]
        ###############################
        if down[0] < up[0]:
            down = down[1:]
        
        #############################
        if (up.shape > down.shape) or (up.shape < down.shape):
            size = np.min([up.shape,down.shape])
            up = up[:size]
            down = down[:size]
        C = np.vstack((up,down))
        ep=np.unique(epochs)
        Density=list()
        for (p1,p2) in C.T:
            idxx=np.where(np.logical_and(ep[p1]< spindles.Onset.values, spindles.Onset.values <ep[p2]))
            if idxx[0].size != 0:
                density=len(spindles.iloc[idxx])/(ep[p2]-ep[p1])
                Density.append(density)
            else:
                Density.append(0)
        Density = np.array(Density)
        average_spindle_density = len(spindles)/t[-1]
        t,p=sts.ttest_1samp(Density,average_spindle_density)
        Above_zero.append(pd.DataFrame({'t':[t],'p':[p],'sub':[file_to_read[:-5]]}))
        
Above_zero=pd.concat(Above_zero)