# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 11:09:05 2016

@author: install
"""

import eegPinelineDesign
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 26})
plt.rc('font', size=26)  
matplotlib.rc('axes', titlesize=22)
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

""" The program will output 2 graphs:
    1. Autodetection graph for spindles
    2. Examples of spindles. """
    

folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
raw = mne.io.read_raw_fif('suj29_l2nap_day2.fif',preload=True,add_eeg_ref=False)
raw2=mne.io.read_raw_fif('suj29_l2nap_day2.fif',preload=True,add_eeg_ref=False)
raw.filter(11,16)
channelList = ['F3','F4','C3','C4','O1','O2']
raw.pick_channels(channelList)
raw2.pick_channels(channelList)
##############################################################################
file_to_read='suj29_l2nap_day2.vhdr'
#fig,(ax1,ax2,ax3)=plt.subplots(figsize=(40,40),nrows=3,sharex=True)
manual = pd.read_csv('suj29_nap_day2_edited_annotations.txt')
time_find,mean_peak_power,Duration,peak_time,peak_at = eegPinelineDesign.spindle_validation_with_sleep_stage(raw,channelList,manual,moving_window_size=500,threshold=.2,
                                                                                                             syn_channels=3,l_freq=11,h_freq=16,l_bound=0.5,h_bound=2,tol=1)
result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
result['Annotation'] = 'auto spindle'
result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
result = result[result.Onset > 100]
Time_ = result.Onset.values
#ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue')

timePoint1 = np.random.choice(manual.Onset.values,1);timePoint2=min(enumerate(Time_), key=lambda x: abs(x[1]-timePoint1))[1]
epoch_length=10 #10sec
epochs = eegPinelineDesign.make_overlap_windows(raw,epoch_length)
epochs = np.unique(epochs)
#############################################################################
fig_ = plt.figure(figsize=(20,20))
man_exam,time_ = eegPinelineDesign.cut_segments(raw,timePoint1,0,windowsize = 3)
aut_exam,time_2 = eegPinelineDesign.cut_segments(raw,timePoint2,0,windowsize=3)
ax1=plt.subplot(421);ax2=plt.subplot(422,sharey=ax1)
ax1.ticklabel_format(useOffset=False);ax2.ticklabel_format(useOffset=False)
ax1.plot(time_2,aut_exam[0,:],color='red')
ax2.plot(time_,man_exam[0,:],color='blue')
ax1.set(ylabel='$\mu$V',title='manual annotated spindle example')
ax2.set(title='auto annotated spindle example')
ax3=plt.subplot(412);ax4=plt.subplot(413,sharex=ax3,sharey=ax3);
ax5=plt.subplot(4,3,10);ax6=plt.subplot(4,3,11,sharex=ax5,sharey=ax5);ax7=plt.subplot(4,3,12,sharex=ax5,sharey=ax5)
segment = raw._data[0,:]
time_span = np.linspace(0,raw.last_samp/raw.info['sfreq'],len(segment))
ax3.plot(time_span,segment,color='black',alpha=0.5)
red_line1=ax3.axvline(timePoint1,color='red',label='manual spindle')
blue_line1=ax3.axvline(timePoint2,color='blue',label='auto spindle')
ax3.set(ylabel='$\mu$V',title='raw data bandpass 11-16 Hz in channel %s' % raw.info['ch_names'][0],xlim=(0,time_span.max()),xticks=(time_span[::100000]))
ldg=ax3.legend(loc=2,ncol=2,prop={'size':6})
plt.setp(ldg.get_texts(),fontsize=12)
temp_alpha=[];temp_beta=[];temp_spindle=[]
for ii,chan in enumerate(raw.info['ch_names']):
    seg,T = raw[ii,:]
    ax4.plot(T,seg[0,:],alpha=0.3,label=chan)
    target_spindle,alpha_C,DT_C,ASI,activity,ave_activity,psd_delta1,psd_delta2,psd_theta,psd_alpha,psd_beta,psd_gamma,slow_spindle,fast_spindle,slow_range,fast_range,_= eegPinelineDesign.epoch_activity(raw2,picks=ii,l=11,h=16)
    ax5.plot(epochs[1:-1],np.array(psd_alpha).mean(1),alpha=0.3)
    temp_alpha.append(np.array(psd_alpha).mean(1))
    ax6.plot(epochs[1:-1],np.array(psd_beta).mean(1),alpha=0.3)
    temp_beta.append(np.array(psd_beta).mean(1))
    ax7.plot(epochs[1:-1],np.array(target_spindle).mean(1),alpha=0.3)
    temp_spindle.append(np.array(target_spindle).mean(1))
ax5.plot(epochs[1:-1],np.array(temp_alpha).mean(0),alpha=1.,color='black')
ax6.plot(epochs[1:-1],np.array(temp_beta).mean(0),alpha=1.,color='black')
ax7.plot(epochs[1:-1],np.array(temp_spindle).mean(0),alpha=1.,color='black')
ax4.set(ylabel='$\mu$V',title='raw data bandpass 11-16 Hz in 6 channels')
red_line2=ax4.axvline(timePoint1,color='red')
red_line2=ax4.axvline(timePoint2,color='blue')
ldg=ax4.legend(loc=2,ncol=6,prop={'size':6})
plt.setp(ldg.get_texts(),fontsize=12)
ax5.set(xlabel='time in seconds',ylabel='Power',title='alpha (8-12 Hz)')
ax6.set(xlabel='time in seconds',title='beta (12-20 Hz)')
ax7.set(xlabel='time in seconds',title='spindle (11 - 16 Hz)')
red_line3=ax5.axvline(timePoint1,color='red')
blue_line3=ax5.axvline(timePoint2,color='blue')
red_line4=ax6.axvline(timePoint1,color='red')
blue_line4=ax6.axvline(timePoint2,color='blue')
red_line5=ax7.axvline(timePoint1,color='red')
blue_line5=ax7.axvline(timePoint2,color='blue')
ax3.set(xlim=(min(timePoint1-10,timePoint2-10),max(timePoint1+10,timePoint2+10)))
ax5.set(xlim=(min(timePoint1-10,timePoint2-10),max(timePoint1+10,timePoint2+10)))
fig_.tight_layout()
fig_.savefig('example and spindle comparison.png',)
