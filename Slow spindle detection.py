# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 10:26:14 2016

@author: asus-task
"""
import eegPinelineDesign
import numpy as np
import mne
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd




folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33)))
had = False
for idx in [29,30,32]: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if str(idx) in folder_to_look]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    print(list_file_to_read);low=1;high=200
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
        raw.filter(12.5,14.5)
        raw.filter(10,12)
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at = eegPinelineDesign.get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,moving_window_size=200,threshold=.9,syn_channels=3,l_freq=10,h_freq=12)
        
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
        result = result[result.Onset > 100]
        Time_ = result.Onset.values
        ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue')
        fileName = file_to_read[:-5] + '_slow_spindle.csv'
        result.to_csv(fileName,spe=',',encoding='utf-8',index=False)
        pic_fileName = fileName[:-4] + 'slow_spindle.png'
        plt.savefig(pic_fileName)