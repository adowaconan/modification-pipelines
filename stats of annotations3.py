# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:50:56 2016

@author: install
"""

import eegPinelineDesign
import pandas as pd
import numpy as np
import mne
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
chan_dict={'Ch56': 'TP8', 'Ch61': 'F6', 'Ch3': 'F3', 'Ch45': 'P1', 'Ch14': 'P3', 
           'Ch41': 'C1', 'Ch1': 'Fp1', 'Ch46': 'P5', 'Ch7': 'FC1', 'Ch37': 'F5', 
           'Ch21': 'TP10', 'Ch8': 'C3', 'Ch11': 'CP5', 'Ch28': 'FC6', 'Ch17': 'Oz', 
           'Ch39': 'FC3', 'Ch38': 'FT7', 'Ch58': 'C2', 'Ch33': 'AF7', 'Ch48': 'PO3', 
           'Ch9': 'T7', 'Ch49': 'POz', 'Ch2': 'Fz', 'Ch15': 'P7', 'Ch20': 'P8', 
           'Ch60': 'FT8', 'Ch57': 'C6', 'Ch32': 'Fp2', 'Ch29': 'FC2', 'Ch59': 'FC4', 
           'Ch35': 'AFz', 'Ch44': 'CP3', 'Ch47': 'PO7', 'Ch30': 'F4', 'Ch62': 'F2', 
           'Ch4': 'F7', 'Ch24': 'Cz', 'Ch31': 'F8', 'Ch64': 'ROc', 'Ch23': 'CP2', 
           'Ch25': 'C4', 'Ch40': 'FCz', 'Ch53': 'P2', 'Ch19': 'P4', 'Ch27': 'FT10', 
           'Ch50': 'PO4', 'Ch18': 'O2', 'Ch55': 'CP4', 'Ch6': 'FC5', 'Ch12': 'CP1', 
           'Ch16': 'O1', 'Ch52': 'P6', 'Ch5': 'FT9', 'Ch42': 'C5', 'Ch36': 'F1', 
           'Ch26': 'T8', 'Ch51': 'PO8', 'Ch34': 'AF3', 'Ch22': 'CP6', 'Ch54': 'CPz', 
           'Ch13': 'Pz', 'Ch63': 'LOc', 'Ch43': 'TP7'}
had = True
current_working_folder=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set\\')
chanName=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz',
          'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 
          'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 
          'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 
          'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 
          'F2', 'LOc', 'ROc', 'Aux1', 'STI 014']
"""
This script is to correct old data using ICA
"""
subjectList=[5,6,8,9,10]
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files) and ('suj%d'%idx in files)]
    print(list_file_to_read);low=1;high=200
    for file_to_read in list_file_to_read:
        if had:
            raw = mne.io.read_raw_brainvision('D:\\NING - spindle\\training set\\'+file_to_read,preload=True,scale=1e6)#[:-5] + '.fif',add_eeg_ref=False
        else:
            try:
                raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=80,eogReject=300,n_ch=-2)
            except:
                try:
                    raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=120,eogReject=300,n_ch=-2)
                except:
                    try:
                        raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=160,eogReject=300,n_ch=-2)
                    except:
                        raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=200,eogReject=300,n_ch=-2)
            raw.save(file_to_read[:-5] + '.fif',overwrite=True)
        try:
            raw.rename_channels(chan_dict)
        except:
            pass
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        l_freq=12.5;h_freq=14.5
        raw.filter(l_freq,h_freq)
        threshold=0.9;moving_window_size=200;l_bound=0.5;h_bound=2;syn_channels=5
        time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at = eegPinelineDesign.get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,
                                                                                                                                         moving_window_size=moving_window_size,
                                                                                                                                         threshold=threshold,
                                                                                                                                         syn_channels=syn_channels,
                                                                                                                                         l_freq=l_freq,
                                                                                                                                         h_freq=h_freq,
                                                                                                                                         tol=1.9)
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