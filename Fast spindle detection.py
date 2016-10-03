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
had = True
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if str(idx) in folder_to_look]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    print(list_file_to_read);
    for file_to_read in list_file_to_read:
        if had:
            raw = mne.io.read_raw_fif(file_to_read[:-5] + '.fif',preload=True,add_eeg_ref=False)
        else:
            try:
                raw = eegPinelineDesign.load_data(file_to_read,low_frequency=.1,high_frequency=50,eegReject=80,eogReject=120,n_ch=-2)
            except:
                try:
                    raw = eegPinelineDesign.load_data(file_to_read,low_frequency=.1,high_frequency=50,eegReject=100,eogReject=150,n_ch=-2)
                except:
                    try:
                        raw = eegPinelineDesign.load_data(file_to_read,low_frequency=.1,high_frequency=50,eegReject=120,eogReject=180,n_ch=-2)
                    except:
                        raw = eegPinelineDesign.load_data(file_to_read,low_frequency=.1,high_frequency=50,eegReject=150,eogReject=200,n_ch=-2)
            raw.save(file_to_read[:-5] + '.fif',overwrite=True)
        #raw = mne.io.read_raw_fif(file_to_read[:-5] + '.fif',preload=True,add_eeg_ref=False)
        l_freq=12.5;h_freq=14.5
        raw.filter(l_freq,h_freq)
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        threshold=0.9;moving_window_size=200;l_bound=0.5;h_bound=2;syn_channels=4
        mul=threshold
        time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at = eegPinelineDesign.get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,
                                                                                                                                         moving_window_size=moving_window_size,
                                                                                                                                         threshold=threshold,
                                                                                                                                         syn_channels=syn_channels,
                                                                                                                                         l_freq=l_freq,
                                                                                                                                         h_freq=h_freq)
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
        result = result[result.Onset > 100]
        Time_ = result.Onset.values
        ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue')
        fileName = file_to_read[:-5] + '_fast_spindle.csv'
        result.to_csv(fileName,spe=',',encoding='utf-8',index=False)
        pic_fileName = fileName[:-4] + 'fast_spindle.png'
        fig.savefig(pic_fileName)
        plt.close(fig)