# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:36:48 2017

@author: install
"""


import mne
import eegPinelineDesign
import numpy as np
import pandas as pd
import pickle
from eegPinelineDesign import sample_data
from eegPinelineDesign import stage_check,sampling_FA_MISS_CR


file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=350;threshold=0.6;syn_channel=3
l,h = (8,16);


# over thresholds 
windowSize=350;threshold=0.6;syn_channel=3
with_sleep_stage,without_sleep_stage,wavelet={},{},{}
for threshold in np.arange(0.05,1.05,0.1):
    temp_with_sleep_stage,temp_without_sleep_stage,temp_wavelet={},{},{};
    with_stage_samples,with_stage_label=[],[]
    without_stage_samples,without_stage_label=[],[]
    with_stage_wavelet_samples,with_stage_wavelet_label=[],[]
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
            ######## get one done #####with sleep stage info#####
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw,
                                                                                                     channelList,file,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.5,
                                                                                                     h_bound=3.0,tol=1)
            
            ###Taking out the first 100 seconds and the last 100 seconds###        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            
            
            
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            temp_with_sleep_stage[sub+day]=[manual_labels,auto_labels,discritized_time_intervals]
            comparedRsult = manual_labels - auto_labels
            with_stage_samples,with_stage_label = sampling_FA_MISS_CR(comparedRsult,manual_labels,raw,annotation,
                                                                      discritized_time_intervals,
                                                                      with_stage_samples,with_stage_label,old)
            
            ###################### get the other one done #####without sleep stage info######
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_step1(raw,
                                                                                                     channelList,file,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.5,
                                                                                                     h_bound=3.0,tol=1)
            
            ###Taking out the first 100 seconds and the last 100 seconds###        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            
            
            
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            temp_without_sleep_stage[sub+day]=[manual_labels,auto_labels,discritized_time_intervals]

            comparedRsult = manual_labels - auto_labels
            without_stage_samples,without_stage_label = sampling_FA_MISS_CR(comparedRsult,manual_labels,raw,annotation,
                                                                      discritized_time_intervals,
                                                                      without_stage_samples,
                                                                      without_stage_label,old)
            
            
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage_after_wavelet_transform(raw,
                                                                                                     channelList,file,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.5,
                                                                                                     h_bound=3.0,tol=1)
            
            ###Taking out the first 100 seconds and the last 100 seconds###        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            
            
            
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            temp_wavelet[sub+day]=[manual_labels,auto_labels,discritized_time_intervals]
            comparedRsult = manual_labels - auto_labels
            with_stage_wavelet_samples,with_stage_wavelet_label = sampling_FA_MISS_CR(comparedRsult,manual_labels,raw,annotation,
                                                                      discritized_time_intervals,
                                                                      with_stage_wavelet_samples,with_stage_wavelet_label,old)
            
        else:
            print(sub+day+'no annotation')
    with_sleep_stage[threshold]=temp_with_sleep_stage;
    without_sleep_stage[threshold]=temp_without_sleep_stage
    wavelet[threshold]=temp_wavelet

    pickle.dump(with_stage_samples,open("temp_data\\%.2f_samples_with.p"%threshold,"wb"))
    pickle.dump(with_stage_label,  open("temp_data\\%.2f_labels_with.p" %threshold,"wb"))
    pickle.dump(without_stage_samples,open("temp_data\\%.2f_samples_without.p"%threshold,"wb"))
    pickle.dump(without_stage_label,  open("temp_data\\%.2f_labels_without.p" %threshold,"wb"))
    pickle.dump(with_stage_wavelet_samples,open("temp_data\\%.2f_samples_with_wavelet.p"%threshold,"wb"))
    pickle.dump(with_stage_wavelet_label,open("temp_data\\%.2f_labels_with_wavelet.p"%threshold,"wb"))

over_threshold={'with':with_sleep_stage,'without':without_sleep_stage}
pickle.dump( over_threshold, open( "over_threshold.p", "wb" ) )

"""
# over tolerance
windowSize=350;threshold=0.6;syn_channel=3
samples={};label={}
with_sleep_stage={};without_sleep_stage={};
for tol in np.arange(0.5,3,0.05):
    temp_with_sleep_stage={};temp_without_sleep_stage={};
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
            ######## get one done #################
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw,
                                                                                                     channelList,file,annotation,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=3.5,tol=tol)
            
            ###Taking out the first 100 seconds and the last 100 seconds###        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            temp_with_sleep_stage[sub+day]=[manual_labels,auto_labels,discritized_time_intervals]


            ###################### get the other one done ##################
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_step1(raw,
                                                                                                     channelList,file,
                                                                                                     moving_window_size=windowSize,
                                                                                                     threshold=threshold,
                                                                                                     syn_channels=syn_channel,
                                                                                                     l_freq=l,
                                                                                                     h_freq=h,
                                                                                                     l_bound=0.55,
                                                                                                     h_bound=3.5,tol=tol)
            
            ###Taking out the first 100 seconds and the last 100 seconds###        
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
            result = result[result.Onset > 100]
            raw.close()
            ################## end of eeg data part ########################
            anno = annotation[annotation.Annotation == 'spindle']['Onset']
            gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
            manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
            auto_labels,discritized_time_intervals = eegPinelineDesign.discritized_onset_label_auto(raw,result,3)
            temp_without_sleep_stage[sub+day]=[manual_labels,auto_labels,discritized_time_intervals]
            
            comparedRsult = manual_labels - auto_labels
            idx_hit = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 1)))[0]
            idx_CR  = np.where(np.logical_and((comparedRsult == 0),(manual_labels == 0)))[0]
            idx_miss= np.where(comparedRsult == 1)[0]
            idx_FA  = np.where(comparedRsult == -1)[0]
            raw_data, time = raw[:,100*raw.info['sfreq']:-100*raw.info['sfreq']]
            stages = annotation[annotation.Annotation.apply(stage_check)]
    
            On = stages[::2];Off = stages[1::2]
            stage_on_off = list(zip(On.Onset.values, Off.Onset.values))
            if abs(np.diff(stage_on_off[0]) - 30) < 2:
                pass
            else:
                On = stages[1::2];Off = stages[::2]
                stage_on_off = list(zip(On.Onset.values[1:], Off.Onset.values[2:]))
    
            for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_miss]):
                a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='miss', old=old)
                
                if len(a) > 0:
                    a = a.tolist()
                    #print(len(a))
                    samples[tol].append(a)
                    label[tol].append(c)
    
            for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_FA]):
    
                a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='fa', old=old)
                if len(a) > 0:
                    a = a.tolist()
                    #print(len(a))
                    samples[tol].append(a)
                    label[tol].append(c)
            b = abs(len(idx_miss) - len(idx_FA))
            for jj, (time_interval_1,time_interval_2) in enumerate(discritized_time_intervals[idx_CR][:b]):
                a,c = sample_data(time_interval_1, time_interval_2, raw, raw_data, stage_on_off, key='cr', old=old)
                if len(a) > 0:
                    a = a.tolist()
                    #print(len(a))
                    samples[tol].append(a)
                    label[tol].append(c)
            
        else:
            print(sub+day+'no annotation')
    with_sleep_stage[tol]=temp_with_sleep_stage;without_sleep_stage[tol]=temp_without_sleep_stage


over_tol={'with':with_sleep_stage,'without':without_sleep_stage}
pickle.dump( over_tol, open( "over_tol.p", "wb" ) )
"""