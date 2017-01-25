# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:55:53 2017

@author: install
"""

import mne
import eegPinelineDesign
from sklearn.metrics import matthews_corrcoef,brier_score_loss,roc_curve,average_precision_score,precision_recall_fscore_support
import numpy as np
import pandas as pd
import pickle


file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=500;threshold=0.85;syn_channel=3
l,h = (11,16);samples=[];label=[]


# over step size
with_sleep_stage=[];without_sleep_stage=[]
for windowSize in [250,500,1000,1250,1500,2000]:
    temp_with_sleep_stage=[];temp_without_sleep_stage=[]
    for file in list_file_to_read:
        sub = file.split('_')[0]
        if int(sub[3:]) >= 11:
            day = file.split('_')[2][:4]
            old = False
        else:
            day = file.split('_')[1]
            old = True
        
                      
        
        
        annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
        if len(annotation_file) != 0:
            print('gathering...................')
            annotation = pd.read_csv(annotation_file[0])
            ################### eeg data part ###########################
            raw = mne.io.read_raw_fif(file,preload=True)
            if old:
                pass
            else:
                raw.resample(500, npad="auto") # down sampling Karen's data
            raw.pick_channels(channelList)
            raw.filter(l,h)
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw,
                                                                                                     channelList,file,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=2.2,tol=1)
            
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            Time_ = result.Onset.values
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
            fpr, tpr = fpr[1], tpr[1]
            temp_with_sleep_stage.append([fpr,tpr])
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            #print(len(idx_hit),len(idx_CR),len(idx_miss),len(idx_FA))
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_step1(raw,
                                                                                                     channelList,file,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=2.2,tol=1)
            
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            Time_ = result.Onset.values
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
            fpr, tpr = fpr[1], tpr[1]
            temp_without_sleep_stage.append([fpr,tpr])
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            #print(len(idx_hit),len(idx_CR),len(idx_miss),len(idx_FA))
        else:
            print(sub+day+'no annotation')
    with_sleep_stage.append(temp_with_sleep_stage);without_sleep_stage.append(temp_without_sleep_stage)
over_step_size={'with':with_sleep_stage,'without':without_sleep_stage}
pickle.dump( over_step_size, open( "over_step_size.p", "wb" ) )

# over thresholds 
with_sleep_stage=[];without_sleep_stage=[]
for threshold in np.arange(0.1,0.95,0.02):
    temp_with_sleep_stage=[];temp_without_sleep_stage=[]
    for file in list_file_to_read:
        sub = file.split('_')[0]
        if int(sub[3:]) >= 11:
            day = file.split('_')[2][:4]
            old = False
        else:
            day = file.split('_')[1]
            old = True
        
                      
        
        
        annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
        if len(annotation_file) != 0:
            print('gathering...................')
            annotation = pd.read_csv(annotation_file[0])
            ################### eeg data part ###########################
            raw = mne.io.read_raw_fif(file,preload=True)
            if old:
                pass
            else:
                raw.resample(500, npad="auto") # down sampling Karen's data
            raw.pick_channels(channelList)
            raw.filter(l,h)
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw,
                                                                                                     channelList,file,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=2.2,tol=1)
            
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            Time_ = result.Onset.values
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
            fpr, tpr = fpr[1], tpr[1]
            temp_with_sleep_stage.append([fpr,tpr])
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            #print(len(idx_hit),len(idx_CR),len(idx_miss),len(idx_FA))
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_step1(raw,
                                                                                                     channelList,file,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=2.2,tol=1)
            
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            Time_ = result.Onset.values
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
            fpr, tpr = fpr[1], tpr[1]
            temp_without_sleep_stage.append([fpr,tpr])
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            #print(len(idx_hit),len(idx_CR),len(idx_miss),len(idx_FA))
        else:
            print(sub+day+'no annotation')
    with_sleep_stage.append(temp_with_sleep_stage);without_sleep_stage.append(temp_without_sleep_stage)
over_threshold={'with':with_sleep_stage,'without':without_sleep_stage}
pickle.dump( over_threshold, open( "over_threshold.p", "wb" ) )



# over syn_channel 
with_sleep_stage=[];without_sleep_stage=[]
for syn_channel in [1,2,3,4,5,6]:
    temp_with_sleep_stage=[];temp_without_sleep_stage=[]
    for file in list_file_to_read:
        sub = file.split('_')[0]
        if int(sub[3:]) >= 11:
            day = file.split('_')[2][:4]
            old = False
        else:
            day = file.split('_')[1]
            old = True
        
                      
        
        
        annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
        if len(annotation_file) != 0:
            print('gathering...................')
            annotation = pd.read_csv(annotation_file[0])
            ################### eeg data part ###########################
            raw = mne.io.read_raw_fif(file,preload=True)
            if old:
                pass
            else:
                raw.resample(500, npad="auto") # down sampling Karen's data
            raw.pick_channels(channelList)
            raw.filter(l,h)
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw,
                                                                                                     channelList,file,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=2.2,tol=1)
            
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            Time_ = result.Onset.values
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
            fpr, tpr = fpr[1], tpr[1]
            temp_with_sleep_stage.append([fpr,tpr])
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            #print(len(idx_hit),len(idx_CR),len(idx_miss),len(idx_FA))
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_step1(raw,
                                                                                                     channelList,file,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=2.2,tol=1)
            
            """Taking out the first 100 seconds and the last 100 seconds"""        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            Time_ = result.Onset.values
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
            fpr, tpr = fpr[1], tpr[1]
            temp_without_sleep_stage.append([fpr,tpr])
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            #print(len(idx_hit),len(idx_CR),len(idx_miss),len(idx_FA))
        else:
            print(sub+day+'no annotation')
    with_sleep_stage.append(temp_with_sleep_stage);without_sleep_stage.append(temp_without_sleep_stage)
over_channel={'with':with_sleep_stage,'without':without_sleep_stage}
pickle.dump( over_channel, open( "over_channel.p", "wb" ) )