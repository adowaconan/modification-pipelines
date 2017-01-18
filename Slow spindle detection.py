# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:26:14 2016

@author: asus-task
"""
import eegPinelineDesign
from eegPinelineDesign import *
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=12); matplotlib.rc('axes', titlesize=12) 
import warnings
warnings.filterwarnings("ignore")
import pandas as pd



All_result = {}
folderList = change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33))) # manually change the range, the second number is the position after the last stop
had = True # meaning I have done the preprocessing and ICA correction
for idx in subjectList: 
    
    folder_ = [folder_to_look for folder_to_look in folderList if str(idx) in folder_to_look]
    current_working_folder = change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    print(list_file_to_read);
    for file_to_read in list_file_to_read:
        if had:
            raw = mne.io.read_raw_fif(file_to_read[:-5] + '.fif',preload=True,add_eeg_ref=False)
        else:
            try:
                raw = eegPinelineDesign.load_data(file_to_read,low_frequency=1,high_frequency=50,eegReject=180,eogReject=300,n_ch=64)
            except:
                try:
                    raw = eegPinelineDesign.load_data(file_to_read,low_frequency=1,high_frequency=50,eegReject=240,eogReject=300,n_ch=64)
                except:
                    try:
                        raw = eegPinelineDesign.load_data(file_to_read,low_frequency=1,high_frequency=50,eegReject=300,eogReject=300,n_ch=64)
                    except:
                        raw = eegPinelineDesign.load_data(file_to_read,low_frequency=1,high_frequency=50,eegReject=360,eogReject=300,n_ch=64)
            raw.save(file_to_read[:-5] + '.fif',overwrite=True)
        
        l_freq=10;h_freq=12 # this is how we define a slpw spindle
        raw.filter(l_freq,h_freq)
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        """Setting parameters"""
        threshold=0.8;moving_window_size=2000;syn_channels=3;
        l_bound=0.5;h_bound=2.3; # not using in the example, they are parameters could be reset in the function
        
        
        time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at = eegPinelineDesign.get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,
                                                                                                                                         moving_window_size=moving_window_size,
                                                                                                                                         threshold=threshold,
                                                                                                                                         syn_channels=syn_channels,
                                                                                                                                         l_freq=l_freq,
                                                                                                                                         h_freq=h_freq)
        """Taking out the first 100 seconds and the last 100 seconds"""        
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
        result = result[result.Onset > 100]
        
        
        fileName = file_to_read[:-5] + '_slow_spindle.csv'
        #All_result[fileName] = result
        
        
        Time_ = result.Onset.values
        ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue')
        fileName = file_to_read[:-5] + '_slow_spindle.csv'
        result.to_csv(fileName,spe=',',encoding='utf-8',index=False)
        pic_fileName = fileName[:-4] + 'slow_spindle.png'
        fig.savefig(pic_fileName)
        
        plt.close('all')
        