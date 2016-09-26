# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:36:51 2016

@author: install
"""

import eegPinelineDesign
import pandas as pd
import numpy as np
import mne
import re
import pickle
def rescale(M):
    M = np.array(M)
    return (M - M.min())/(M.max() - M.min())+2
results={}

sublist=[13,28,29]
for sub in sublist:
    current_working_folder=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\suj%d'%(sub))
    list_file_to_read = [files for files in current_working_folder if ('txt' in files) and ('nap' in files)]
    for file_to_read in list_file_to_read:
        annotation = pd.read_csv(file_to_read)
        raw_file_to_read =[file for file in current_working_folder if (file_to_read.split('_')[0] in file) and (file_to_read.split('_')[1] in file) and (file_to_read.split('_')[2] in file) and ('fif' in file)]
        raw = mne.io.read_raw_fif(raw_file_to_read,preload=True,add_eeg_ref=False)
        channelList = ['F3','F4','C3','C4','O1','O2']
        raw.pick_channels(channelList)
        _,times=raw[:]
        key=re.compile('Marker',re.IGNORECASE)

        temp=[]
        for row in enumerate(annotation.iterrows()):
            #print(row[1][-1])
            if key.search(row[1][-1][-1]):
                temp.append([row[1][-1][-2],row[1][-1][-1]])
        temp = pd.DataFrame(temp,columns=['Onset','annotation'])

        temp['annotation']=temp.annotation.apply(eegPinelineDesign.recode_annotation)
        #temp.plot(x='Onset',y='annotation')

        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,stim=False)
        result={}
        alpha_C,DT_C,ASI,activity,ave_activity,psd_delta1,psd_delta2,psd_theta,psd_alpha,psd_beta,psd_gamma,slow_spindle,fast_spindle,slow_range,fast_range,epochs = eegPinelineDesign.epoch_activity(raw,picks=picks)
        activity = np.array(activity)
        ave_activity=np.array(ave_activity)
        alpha_C=np.array(alpha_C)
        DT_C=np.array(DT_C)
        ASI = np.array(ASI)
        power_fast_spindle = np.array(fast_spindle)
        My_ASI = (np.log2(rescale(np.array(psd_alpha).mean(1))) + np.log2(rescale(np.array(psd_beta).mean(1)))) / np.log2(rescale(power_fast_spindle.mean(1)))
        result['alpha activity']=alpha_C;result['sum of delta and theta']=DT_C
        result['activity across 6 bands']=activity;result['delta 0-2']=psd_delta1
        result['delta 2-4']=psd_delta2;result['theta']=psd_theta;result['alpha']=psd_alpha
        result['beta']=psd_beta;result['gamma']=psd_gamma
        result['beta_mean']=np.array(psd_beta).mean(1)
        result['gamma_std']=np.array(psd_gamma).std(1)
        result['activity']=np.array(ave_activity)
        result['slow']=np.array(slow_spindle)
        result['fast']=np.array(fast_spindle)
        result['my ASI']=np.array(My_ASI)
        result['epochs']=np.unique(epochs)
        #pickle.dump( result, open( 'suj13_l2nap_day2_fast_spindle.p', "wb" ) )
        results[file_to_read[:-16]]={'annotation':annotation,'result':result}

pickle.dump(results,open('sleep annotation.p','wb'))