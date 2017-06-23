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
matplotlib.rc('axes', titlesize=16)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
import pickle

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

spindle_files = [f for f in os.listdir('D:\\NING - spindle\\DatabaseSpindles\\') if ('scoring1' in f)]
ss = []
for f in spindle_files:
    spindle = np.loadtxt('D:\\NING - spindle\\DatabaseSpindles\\'+f,skiprows=1)
    annotation = pd.DataFrame({'Onset':spindle[:,0],'duration':spindle[:,1]})
    ss.append(annotation)
ss = pd.concat(ss)
fif_files = [f for f in os.listdir('D:\\NING - spindle\\DatabaseSpindles\\') if ('fif' in f)]
for f in fif_files[:1]:
    raw_ = mne.io.read_raw_fif('D:\\NING - spindle\\DatabaseSpindles\\'+f,preload=True,add_eeg_ref=False)
    raw2_ = mne.io.read_raw_fif('D:\\NING - spindle\\DatabaseSpindles\\'+f,preload=True,add_eeg_ref=False)
raw_.filter(11,16)
#result,auc,auto,manual,times,auto_proba = new_data_pipeline(raw2,
#                            'D:\\NING - spindle\\DatabaseSpindles\\Visual_scoring1_excerpt1.txt',
#                            'D:\\NING - spindle\\DatabaseSpindles\\Hypnogram_excerpt1.txt',
#                            lower_threshold=.48,higher_threshold=3.48)
#dreams= [result,auc,auto,manual,times,auto_proba]
#pickle.dump(dreams, open('dreams 1.p','wb'))
dreams = pickle.load(open('dreams 1.p','rb'))
dpoint = 637.2200	; dduration = 1.0200
dpoint_auto = 875.050000; dduration_atuo = 1.890000

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
#np.save('auto durations.npy',durations)

channelList = ['F3','F4','C3','C4','O1','O2']
fig, axes = plt.subplots(nrows=2,ncols=3,figsize=(16,8))
ax = axes[1]
for ii,timepoint in enumerate([timePoint3,timePoint5]):
    seg, time = eegPinelineDesign.cut_segments(raw2,timepoint+1,0)
    ax[ii].plot(time,seg[0,:],label=channelList[0])
    ax[ii].axvline(timepoint,color='red',lw=2,)#label='Onset: %.2f sec'%(timepoint))
    ax[ii].axvspan(timepoint-0.5,timepoint+1.5,color='red',alpha=0.5,)#label='Default duration: 2 sec')
#    ax[ii].set(xticks = [timepoint],xticklabels=[timepoint],xlabel='')
    ax[ii].set(xlabel='')
    ax[ii].legend(loc='best',prop={'size':12})
    ax[ii].set_title('Manual detected spindle example%d,\n raw signal (0.1 - 50 Hz)'%(ii+1),fontweight='bold',fontsize=15) 
    #ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
ax[0].set_ylabel('$\mu$V',fontweight='bold')

seg, time = eegPinelineDesign.cut_segments(raw2_, dpoint,0)
ax[2].plot(time, seg[0,:],label='Cz')
ax[2].axvspan(dpoint-dduration/2,dpoint+dduration/2,color='red',alpha=0.5,)
#  label='Onest: %d sec\nDuration: %.2f sec'%(dpoint-dduration/2,dduration/2))
#ax[2].set(xticks=[dpoint-dduration/2],xticklabels=['%d + %.2f'%(dpoint-dduration/2,dduration/2)],xlabel='')
ax[2].set(xlabel='')
ax[2].legend(loc='best',prop={'size':12})
ax[2].set_title('Manual detected spindle example3,\n DREAMS dataset (0.1 - 50 Hz)',fontweight='bold',fontsize=15) 

ax = axes[0]
for ii,timepoint in enumerate([timePoint3,timePoint5]):
    for n_ch in range(6):
        seg, time = eegPinelineDesign.cut_segments(raw,timepoint+1,n_ch)
        ax[ii].plot(time,seg[0,:],alpha=0.7)#,label=channelList[n_ch])
    ax[ii].axvline(timepoint,color='red',lw=2,label='Onset: %.2f sec'%(timepoint))
    ax[ii].axvspan(timepoint-0.5,timepoint+1.5,color='red',alpha=0.5,label='Defined duration: 2 sec')
#    ax[ii].set(xticks = [timepoint],xticklabels=[timepoint],xlabel='')
    ax[ii].set(xlabel='')
    ax[ii].set_xlabel('Time (Sec)',fontweight='bold')
    ax[ii].legend(loc='best',prop={'size':12})
    ax[ii].set_title('Manual detected spindle example%d,\n11 - 16 Hz'%(ii+1),fontweight='bold',fontsize=15)
ax[0].set_ylabel('$\mu$V',fontweight='bold')
# 
seg, time = eegPinelineDesign.cut_segments(raw_, dpoint,0)
ax[2].plot(time, seg[0,:],label='Cz')
ax[2].axvspan(dpoint-dduration/2,dpoint+dduration/2,color='red',alpha=0.5,
  label='Onest: %d sec\nDuration: %.2f sec'%(dpoint-dduration/2,dduration/2))
#ax[2].set(xticks=[dpoint-dduration/2],xticklabels=['%d + %.2f'%(dpoint-dduration/2,dduration/2)],xlabel='')
ax[2].set(xlabel='')
ax[2].set_xlabel('Time (Sec)',fontweight='bold')
ax[2].legend(loc='best',ncol=2,prop={'size':12})
ax[2].set_title('Manual detected spindle example3,\nDREAMS dataset 11 - 16 Hz',fontweight='bold',fontsize=15)
fig.tight_layout()
fig.savefig('example of manually annotated spindles.png',dpi=300)
#############################################################################################
fig = plt.figure(figsize=(16,12))
ax = np.array([fig.add_subplot(334),fig.add_subplot(335),fig.add_subplot(336)])
for ii,timepoint in enumerate([timePoint4,timePoint6]):
    seg, time = eegPinelineDesign.cut_segments(raw2,timepoint,0)
    c = ax[ii]
    c.plot(time,seg[0,:],label=channelList[0])
    duration = Duration[np.where(time_find == timepoint)]
    c.axvspan(timepoint-duration, timepoint+duration, color='red',alpha=0.5,)
#    c.set(xticks = [timepoint-duration],xlabel='',
#      xticklabels=['%.2f + %.2f'%(timepoint,duration)])
    c.set(xlabel='')
    for tick in ax[ii].xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
    ax[ii].set_xlabel('Time (Sec)',fontweight='bold',fontsize=20)
    c.legend(loc='best',prop={'size':12})
    ax[ii].set_title('Auto detected spindle example%d,\n raw signal (0.1 - 50 Hz)'%(ii+1),fontweight='bold',fontsize=15)
ax[0].set_ylabel('$\mu$V',fontweight='bold')
 
seg, time = eegPinelineDesign.cut_segments(raw2_, dpoint_auto,0)
ax[2].plot(time, seg[0,:],label='Cz')
ax[2].axvspan(dpoint_auto-dduration_atuo/2,dpoint_auto+dduration_atuo/2,color='red',alpha=0.5)
#ax[2].set(xticks=[dpoint_auto-dduration_atuo/2],xticklabels=['%d + %.2f'%(dpoint_auto-dduration_atuo/2,dduration_atuo/2)],xlabel='')
ax[2].set_xlabel('Time (Sec)',fontweight='bold',fontsize=20)
ax[2].legend(loc='best',prop={'size':12})
ax[2].set_title('Auto detected spindle example3,\n DREAMS dataset (0.1 - 50 Hz)',fontweight='bold',fontsize=15)
for tick in ax[2].xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 

ax = np.array([fig.add_subplot(331),fig.add_subplot(332),fig.add_subplot(333)])
for ii,timepoint in enumerate([timePoint4,timePoint6]):
    for n_ch in range(6):
        seg, time = eegPinelineDesign.cut_segments(raw,timepoint,n_ch)
        ax[ii].plot(time,seg[0,:],alpha=0.7)#,label=channelList[n_ch])
    duration = Duration[np.where(time_find == timepoint)]
    ax[ii].axvline(timepoint,color='red',alpha=1.,label='Marked Peak: %.2f sec'%(timepoint))
    ax[ii].axvspan(timepoint-duration, timepoint+duration, color='red',alpha=0.5,
      label='Duration: %.2f sec'%(duration))
#    ax[ii].set(xticks = [timepoint-duration],
#      xticklabels=['%.2f + %.2f'%(timepoint,duration)])
    ax[ii].set_xlabel('',)
    for tick in ax[ii].xaxis.get_major_ticks():
                tick.label.set_fontsize(20) 
#    ax[ii].set_xlabel('Time (Sec)',fontweight='bold',fontsize=20)
    ax[ii].legend(loc='best',prop={'size':12})
    ax[ii].set_title('Auto detected spindle example%d,\n11 - 16 Hz'%(ii+1),fontweight='bold',fontsize=15)
ax[0].set_ylabel('$\mu$V',fontweight='bold')
    
seg, time = eegPinelineDesign.cut_segments(raw_, dpoint_auto,0)
ax[2].plot(time, seg[0,:])#,label='Cz')
ax[2].axvline(dpoint_auto,color='red',alpha=1.,label='Marked Peak: %.2f sec'%(dpoint_auto))
ax[2].axvspan(dpoint_auto-dduration_atuo/2,dpoint_auto+dduration_atuo/2,color='red',alpha=0.5,
  label='Duration: %.2f sec'%(dduration_atuo))
#ax[2].set(xticks=[dpoint_auto-dduration_atuo/2],xticklabels=['%d + %.2f'%(dpoint_auto-dduration_atuo/2,dduration_atuo/2)],xlabel='')
ax[2].set(xlabel='')
ax[2].legend(loc='best',prop={'size':12})
ax[2].set_title('Auto detected spindle example3,\n DREAMS dataset 11 - 16 Hz',fontweight='bold',fontsize=15)
for tick in ax[2].xaxis.get_major_ticks():
    tick.label.set_fontsize(20) 

ax = fig.add_subplot(313)
#fig, ax = plt.subplots()
durations = np.load('auto durations.npy')
ax.hist(durations,bins=75,color='blue',alpha=0.5,normed=True)
ax.axvline(np.mean(durations),color='black',alpha=1.0,
           label='Average durations (Sec): %.2f +/- %.2f'%(durations.mean(),durations.std()))
ax.axvspan(durations.mean()-durations.std(),
           durations.mean()+durations.std(),color='blue',alpha=0.5)
ax.hist(ss['duration'].values,color='red',bins=75,alpha=0.5,normed=True)
ax.axvline(np.mean(ss['duration']),color='red',alpha=1.0,
                   label='Average durations (Sec)\nDREAMS data: %.2f +/- %.2f'%(ss['duration'].mean(),ss['duration'].std()) ) 
ax.axvspan(ss['duration'].mean()-ss['duration'].std(),
           ss['duration'].mean()+ss['duration'].std(),
           color='red',alpha=0.5,label=None)
ax.set_title('Distribution of durations of auto-detected spindles',fontweight='bold',fontsize=20)
ax.set_ylabel('Normalized \nfrequency', fontweight='bold',fontsize=20)
ax.set_xlabel('Duration (Sec)',fontweight='bold',fontsize=20)  
ax.legend(loc='upper right',prop={'size':16}) 
fig.tight_layout()
fig.savefig('example of auto annotated spindles.png',dpi=300)

