# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 14:44:16 2017

@author: ning
"""

import eegPinelineDesign
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
plt.rc('font', size=16,weight='bold')  
matplotlib.rc('axes', titlesize=22)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
raw = mne.io.read_raw_fif('suj29_l2nap_day2.fif',preload=True,add_eeg_ref=False)
raw2=mne.io.read_raw_fif('suj29_l2nap_day2.fif',preload=True,add_eeg_ref=False)
raw.filter(11,16)
channelList = ['F3','F4','C3','C4','O1','O2']
raw.pick_channels(channelList)
raw2.pick_channels(channelList)
annotation = pd.read_csv('suj29_nap_day2_edited_annotations.txt')
time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,channelList,annotation,moving_window_size=200,
                                        lower_threshold=0.4,
                                        syn_channels=3,l_bound=0.5,h_bound=2,tol=1,higher_threshold=3.5,
                                        front=300,back=100,sleep_stage=True,proba=True,validation_windowsize=3
                                        )
time_find, Duration = np.array(time_find),np.array(Duration)
np.random.seed(12345)
timePoint1 = np.random.choice(annotation.Onset.values,1)[0];timePoint2=min(enumerate(time_find), key=lambda x: abs(x[1]-timePoint1))[1]
timePoint3 = np.random.choice(annotation.Onset.values,1)[0];timePoint4=min(enumerate(time_find), key=lambda x: abs(x[1]-timePoint3))[1]
timePoint5 = np.random.choice(annotation.Onset.values,1)[0];timePoint6=min(enumerate(time_find), key=lambda x: abs(x[1]-timePoint5))[1]

#examples = np.random.choice(annotation.Onset.values,3,replace=False)
#fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(16,8))
#ax = axes[0]
#for ii,timepoint in enumerate(examples):
#    seg, time = eegPinelineDesign.cut_segments(raw2,timepoint+1,0,3)
#    ax[ii].plot(time,seg[0,:],)
#    ax[ii].set(xticklabels='')
#    #ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
#    ax[ii].set_title('Example %d'%(ii+1),fontweight='bold')
#ax[0].set_ylabel('$\mu$V',fontweight='bold')
#ax = axes[1]
#for ii,timepoint in enumerate(examples):
#    seg, time = eegPinelineDesign.cut_segments(raw,timepoint+1,0,3)
#    ax[ii].plot(time,seg[0,:],)
#    ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
#    ax[ii].set(ylim=(-40,40))
#ax[0].set_ylabel('$\mu$V',fontweight='bold')
#fig.savefig('example of manually annotated spindles.png')

fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(16,5))
ax = axes[0]
for ii,timepoint in enumerate([timePoint1,timePoint3,timePoint5]):
    seg, time = eegPinelineDesign.cut_segments(raw2,timepoint+1,0,3)
    
    ax[ii].plot(time,seg[0,:],)
    ax[ii].axvline(timepoint,color='red',lw=2)
    ax[ii].axvspan(timepoint-0.5,timepoint+1.5,color='red',alpha=0.5)
    ax[ii].set(xticks = [timepoint],xticklabels=[timepoint],xlabel='')
    
    
    #ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
ax[0].set_ylabel('$\mu$V',fontweight='bold')
ax[1].set_title('Manual detected spindle examples,\n raw signal (0.1 - 50 Hz)',fontweight='bold',fontsize=12) 
ax = axes[1]
for ii,timepoint in enumerate([timePoint1,timePoint3,timePoint5]):
    seg, time = eegPinelineDesign.cut_segments(raw,timepoint+1,0,3)
    ax[ii].plot(time,seg[0,:],)
    ax[ii].axvline(timepoint,color='red',lw=2)
    ax[ii].axvspan(timepoint-0.5,timepoint+1.5,color='red',alpha=0.5)
    ax[ii].set(xticks = [timepoint],xticklabels=[timepoint],xlabel='')
    ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
ax[0].set_ylabel('$\mu$V',fontweight='bold')
ax[1].set_title('Manual detected spindle examples,\n bandpass filter 11 - 16 Hz',fontweight='bold',fontsize=12) 
fig.tight_layout()
fig.savefig('example of manually annotated spindles.png')

fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(16,5))
ax = axes[0]
for ii,timepoint in enumerate([timePoint2,timePoint4,timePoint6]):
    seg, time = eegPinelineDesign.cut_segments(raw2,timepoint,0,3)
    ax[ii].plot(time,seg[0,:],)
    duration = Duration[np.where(time_find == timepoint)]
    ax[ii].axvspan(timepoint-duration, timepoint+duration, color='red',alpha=0.5)
    ax[ii].set(xticks = [timepoint-duration],xlabel='',
      xticklabels=['%.2f + %.2f'%(timepoint,duration)])
    #ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
#seg, time = eegPinelineDesign.cut_segments(raw,598.837,0,3)
#d = 0.607
#ax[0].axvspan(598.837-d,598.837+d,color='red',alpha=0.5)
#ax[0].set(xticks = [timePoint2-duration,598.837-d],xlabel='',
#      xticklabels=['%.2f + \n%.2f'%(timePoint2,duration),'%.2f + \n%.2f'%(598.837,d)])
ax[0].set_ylabel('$\mu$V',fontweight='bold')
ax[1].set_title('Auto detected spindle examples,\n raw signal (0.1 - 50 Hz)',fontweight='bold',fontsize=12) 
ax = axes[1]
for ii,timepoint in enumerate([timePoint2,timePoint4,timePoint6]):
    seg, time = eegPinelineDesign.cut_segments(raw,timepoint,0,3)
    ax[ii].plot(time,seg[0,:],)
    duration = Duration[np.where(time_find == timepoint)]
    ax[ii].axvspan(timepoint-duration, timepoint+duration, color='red',alpha=0.5)
    ax[ii].set(xticks = [timepoint-duration],
      xticklabels=['%.2f + %.2f'%(timepoint,duration)])
    ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
#seg, time = eegPinelineDesign.cut_segments(raw,598.837,0,3)
#d = 0.607
#ax[0].axvspan(598.837-d,598.837+d,color='red',alpha=0.5)
#ax[0].set(xticks = [timePoint2-duration,598.837-d],xlabel='',
#      xticklabels=['%.2f + \n%.2f'%(timePoint2,duration),'%.2f + \n%.2f'%(598.837,d)])
ax[0].set_ylabel('$\mu$V',fontweight='bold')
ax[1].set_title('Auto detected spindle examples,\n bandpass filter 11 - 16 Hz',fontweight='bold',fontsize=12)        
fig.tight_layout()
fig.savefig('example of auto annotated spindles.png')

