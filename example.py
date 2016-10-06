# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:57:29 2016

@author: install
"""

import eegPinelineDesign
import mne
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, arange
import numpy as np
def plotSpectrum(y,Fs,ax,color='r',alpha=0.2):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y) # length of the signal
    k = arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    idx = np.where(np.logical_and(frq>9,frq<15))
    
    Y = fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
     
    ax.plot(frq[idx],abs(Y)[idx],color=color,alpha=alpha) # plotting the spectrum
    ax.set(xlabel='Freq (Hz)',ylabel='power')
    return frq[idx],abs(Y)[idx]
 
eegPinelineDesign.change_file_directory('D:\\NING - spindle\\suj29')

raw = mne.io.read_raw_fif('suj29_l2nap_day2.fif',preload=True,add_eeg_ref=False)
channelList = ['F3','F4','C3','C4','O1','O2']
raw.pick_channels(channelList)
fastSpindle = pd.read_csv('suj29_l2nap_day2_fast_spindle.csv')
slowSpindle = pd.read_csv('suj29_l2nap_day2_slow_spindle.csv')

idx = 50;window = 5

fastSpindleInstance = fastSpindle['Onset'][48]
slowSpindleInstance = slowSpindle['Onset'][35]


fig = plt.figure(figsize=(20,20))
ax=fig.add_subplot(421)
start,stop=raw.time_as_index([slowSpindleInstance-window/2,slowSpindleInstance+window/2])
s,t = raw[:,start:stop]
for ii in range(6):
    s[ii,:] = s[ii,:] - s[ii,:].mean()
    ax.plot(t,s[ii,:])
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.set(title='slow spindle example',xlabel='time',ylabel='$\mu$V')
ax=fig.add_subplot(423,sharex=ax)
for ii in range(6):
    ax.plot(t,mne.filter.band_pass_filter(s[ii,:],1000,10,12),alpha=0.5)
ax.axvline(slowSpindleInstance,color='k',alpha=1.,linewidth=2)
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.set(title='band pass 10 - 12 Hz',xlim=(1020,1027))
ax0=fig.add_subplot(222)
Y = np.zeros((6,30))
for ii in range(6):
    freq, Y[ii,:]=plotSpectrum(s[ii,:],1000,ax0,)
ax0.plot(freq,Y.mean(0),'k',alpha=1.,linewidth=5)
ax0.set(title='power spectral density')

ax=fig.add_subplot(425)
start,stop=raw.time_as_index([fastSpindleInstance-window/2,fastSpindleInstance+window/2])
s,t = raw[:,start:stop]
for ii in range(6):
    s[ii,:] = s[ii,:] - s[ii,:].mean()
    ax.plot(t,mne.filter.band_pass_filter(s[ii,:],1000,1,4))
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.set(title='fast spindle example',xlabel='time',ylabel='$\mu$V')
ax=fig.add_subplot(427,sharex=ax)
for ii in range(6):
    ax.plot(t,mne.filter.band_pass_filter(s[ii,:],1000,12,14),alpha=0.5)
ax.axvline(fastSpindleInstance,color='k',alpha=1.,linewidth=2)
ax.get_xaxis().get_major_formatter().set_useOffset(False)
ax.set(title='band pass 12 - 14 Hz',xlim=(1020,1027))
ax0=fig.add_subplot(224,sharex=ax0)
Y = np.zeros((6,29))
for ii in range(6):
    freq, Y[ii,:]=plotSpectrum(s[ii,:],1000,ax0,)
ax0.plot(freq,Y.mean(0),'k',alpha=1.,linewidth=5)
ax0.set(title='power spectral density')
plt.tight_layout()