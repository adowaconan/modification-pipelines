# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 12:28:12 2016

@author: install
"""

import eegPinelineDesign
from eegPinelineDesign import regressionline,find_title_peak
from scipy.stats import linregress
import mne
from mne.time_frequency import psd_multitaper
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33)))
result = dict()
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if (str(idx) in folder_to_look) and ('suj' in folder_to_look)]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    for file_to_read in list_file_to_read:    
        raw=mne.io.read_raw_fif(file_to_read[:-5]+'.fif',preload=True,add_eeg_ref=False)
        result[file_to_read[:-5]]=dict()
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        picks = mne.pick_types(raw.info,eeg=True,eog=False)
        raw.filter(1,None,picks=picks,l_trans_bandwidth=0.5)
        fmin=0;fmax=50;epoch_length=30
        epochs = eegPinelineDesign.make_overlap_windows(raw,epoch_length=epoch_length)
        psd_dict=dict()
        for names in channelList:
            psd_dict[names]=[]
        for ii,epch in enumerate(epochs):
            eegPinelineDesign.update_progress(ii,len(epochs))
            psds, freqs = psd_multitaper(raw, low_bias=True, tmin=epch[0], tmax=epch[1],
                             fmin=fmin, fmax=fmax, proj=False, picks=picks,
                             n_jobs=-1)
            psds = 10 * np.log10(psds)
            for jj,names in enumerate(channelList):
                psd_dict[names].append(psds[jj,:])
        psd_by_channel=dict()
        temp_total=[]
        fig=plt.figure(figsize=(30,30))
        ax0=fig.add_subplot(331)
        for jj,names in enumerate(channelList):
            temp_total.append(psd_dict[names])
            temp = np.vstack(psd_dict[names])
            psd_by_channel[names]=[temp.mean(0),temp.std(0)]
            if jj == 0:
                ax = ax0
            else:
                ax=fig.add_subplot(3,3,jj+1,sharex=ax0)
            idxs=np.logical_and(freqs>9.5,freqs<15)
            x = freqs[idxs]
            y = temp.mean(0)[idxs]
            
            yerr = temp.std(0)[idxs]/np.sqrt(len(epochs))
            ax.errorbar(x,y,marker='.',fmt='',color='black',alpha=1.)
            ax.set(title=names,xlabel='frequency',ylabel='power')
            ax.fill_between(x,y-yerr,y+yerr,alpha=0.5,color='red')
            s,intercept,_,_,_ = linregress(x=x,y=y)
            ax.plot(x,regressionline(intercept,s,x))
            
            idx_devia,maxArg=find_title_peak(x,y)
            if maxArg+idx_devia > len(x):
                idx_devia = len(x) - maxArg - 1
            ax.annotate('freq: %.2f Hz,range: %.2f-%.2f'%(x[maxArg],x[maxArg-idx_devia],x[maxArg+idx_devia]),xy=(x[maxArg],y[maxArg]))
            ax.plot(x[maxArg],y[maxArg],'s',color='yellow')
            ax.axvspan(x[maxArg-idx_devia],x[maxArg+idx_devia],color='green',alpha=0.5)
            result[file_to_read[:-5]][names]={'peak':x[maxArg],'range':x[maxArg+idx_devia]-x[maxArg],'x':x,'y':y,'yerr':yerr}
            
        
        psd_total = np.vstack(temp_total)
        ax = fig.add_subplot(3,1,3,sharex=ax0)
        x = freqs[idxs]
        y = psd_total.mean(0)[idxs]
        yerr = psd_total.std(0)[idxs]/np.sqrt(len(epochs))
        ax.errorbar(x,y,marker='.',fmt='',color='k',alpha=1.)
        ax.fill_between(x,y-yerr,y+yerr,alpha=0.5,color='red')
        ax.set(title=file_to_read[:-5],xlabel='frequency',ylabel='power')
        
        
        idx_devia,maxArg=find_title_peak(x,y)
        if maxArg+idx_devia > len(x):
            idx_devia = len(x) - maxArg - 1
        ax.annotate('freq: %.2f Hz'%x[maxArg],xy=(x[maxArg],y[maxArg]))
        ax.plot(x[maxArg],y[maxArg],'s',color='yellow')
        ax.axvspan(x[maxArg-idx_devia],x[maxArg+idx_devia],color='green',alpha=0.5)
        result[file_to_read[:-5]]['mean']={'peak':x[maxArg],'range':x[maxArg+idx_devia]-x[maxArg],'x':x,'y':y,'yerr':yerr}
        
        
        
        fig.tight_layout()
        fileName = file_to_read[:-5] + 'power_spectral.png'
        fig.savefig(fileName)
        plt.close(fig)
"""
import pickle
eegPinelineDesign.change_file_directory('D:\\NING - spindle')
with open( 'total results.p','wb') as h:
    pickle.dump(result,h)
"""