# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:59:32 2017

@author: install
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import eegPinelineDesign
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import matthews_corrcoef,brier_score_loss,roc_curve,average_precision_score,precision_recall_fscore_support
import re
import pandas as pd
import mne

try:
    file_in_fold = eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\allData')
except:
    file_in_fold = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData')
####################################################################################################
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]    
"""Setting parameters"""
had = True
spindle_segment = 3 # define spindle long as 2 seconds for all manual annotations


channelList = ['F3','F4','C3','C4','O1','O2']
moving_window_size=2000;#syn_channels=int(.75 * len(channelList));
l_bound=0.55;h_bound=2.2; # not using in the example, they are parameters could be reset in the function
thresholds = np.arange(0.1,.95,.05);syn_channels = 3
bands = [[10,12],[12,14],[10,14],[10,18],[12,18]]
Predictions = {}

names = 'tn,fp,fn,tp,precision,recall,fbeta_score,tpr,fpr,matthews_corrcoef'.split(',')
for ii,sample in enumerate(list_file_to_read):
    sub = sample.split('_')[0]
    day = sample.split('_')[-1][:4]
    
    
    band_threshold_temp=[]
    for n_band,band in enumerate(bands):
        raw = mne.io.read_raw_fif(sample,preload=True,add_eeg_ref=False)
        l_freq,h_freq = band
        raw.pick_channels(channelList)
        raw.filter(l_freq,h_freq)
        
        print('subject',sub,'day',day)
        threshold_temp = []
        for n_threshold, threshold in enumerate(thresholds):
            print('threshold = %.3f' % threshold)
            time_find,mean_peak_power,Duration,peak_time,peak_at = eegPinelineDesign.spindle_validation_step1(raw,channelList,sample,
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
            threshold_temp.append(result)
        
        band_threshold_temp.append(threshold_temp)
        
    Predictions[str(sub)+'_'+str(day)]=band_threshold_temp

pickle.dump( Predictions, open( "fixed channel.p", "wb" ) )


try:
    annotation_in_fold = eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\allData\\annotation')
except:
    annotation_in_fold = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData\\annotation')
temp = []
for item in annotation_in_fold:
    temp_item = 'annotation/' + item
    temp.append(temp_item)
annotation_in_fold = temp


try:
    file_in_fold = eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\allData')
except:
    file_in_fold = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData')

####################################################################################################
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]    
"""Setting some parameters"""
#had = True
#spindle_segment = 3 # define spindle long as 2 seconds for all manual annotations
#channelList = ['F3','F4','C3','C4','O1','O2']
#moving_window_size=200;#syn_channels=int(.75 * len(channelList));
#l_bound=0.5;h_bound=2; # not using in the example, they are parameters could be reset in the function
#thresholds = np.arange(0.1,.95,.05);syn_channels = 4
#bands = [[10,12],[12,14],[10,14],[10,18],[12,18]]
names = 'tn,fp,fn,tp,precision,recall,fbeta_score,tpr,fpr,matthews_corrcoef'.split(',')
with open('fixed channel.p', 'rb') as handle:
    allResults = pickle.load(handle)
validation = {name:{} for name in names}
validations={name:np.zeros((len(bands),len(thresholds),len(list_file_to_read))) for name in names}
results_in_list={}
for subNum,file in enumerate(list_file_to_read):
    try:
        sub = file.split('_')[0]
        day = file.split('_')[2][:4]
        annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
        #print(file,annotation_file)
        
        raw = mne.io.read_raw_fif(file,preload=False)
        gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
        manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,spindle_segment)
        
        auto_results = allResults[str(sub)+'_'+str(day)]
        temp_band=[]
        for s,band in enumerate(bands):
            result_by_band = auto_results[s]
            temp_threshold=[]
            for t, threshold in enumerate(thresholds):
                result_by_threshold = result_by_band[t]
                
                auto_labels = eegPinelineDesign.discritized_onset_label_auto(raw,result_by_threshold,spindle_segment)
                tn, fp, fn, tp = eegPinelineDesign.confusion_matrix(manual_labels,auto_labels).ravel()
                prec, rec, fbeta, supp=precision_recall_fscore_support(manual_labels,auto_labels,pos_label=None,average='binary')
                fpr,tpr,_ = roc_curve(manual_labels,auto_labels)
                fpr, tpr = fpr[1], tpr[1]
                corr = matthews_corrcoef(manual_labels,auto_labels)
                
                validation['tn'][str(subNum)+'_'+str(s)+'_'+str(t)]=tn
                validation['fp'][str(subNum)+'_'+str(s)+'_'+str(t)]=fp
                validation['fn'][str(subNum)+'_'+str(s)+'_'+str(t)]=fn
                validation['tp'][str(subNum)+'_'+str(s)+'_'+str(t)]=tp
                validation['precision'][str(subNum)+'_'+str(s)+'_'+str(t)]=prec
                validation['recall'][str(subNum)+'_'+str(s)+'_'+str(t)]=rec
                validation['fbeta_score'][str(subNum)+'_'+str(s)+'_'+str(t)]=fbeta
                validation['tpr'][str(subNum)+'_'+str(s)+'_'+str(t)]=tpr
                validation['fpr'][str(subNum)+'_'+str(s)+'_'+str(t)]=fpr
                validation['matthews_corrcoef'][str(subNum)+'_'+str(s)+'_'+str(t)]=corr
                
                #validation['roc_auc'][str(subNum)+'_'+str(s)+'_'+str(t)]=roc_auc
                validations['tn'][s,t,subNum]=tn
                validations['fp'][s,t,subNum]=fp
                validations['fn'][s,t,subNum]=fn
                validations['tp'][s,t,subNum]=tp
                validations['precision'][s,t,subNum]=prec
                validations['recall'][s,t,subNum]=rec
                validations['fbeta_score'][s,t,subNum]=fbeta
                validations['tpr'][s,t,subNum]=tpr
                validations['fpr'][s,t,subNum]=fpr
                validations['matthews_corrcoef'][s,t,subNum]=corr
                #validations['roc_auc'][subNum,s,t]=roc_auc
                print(subNum,s,t,'tn',tn,'fp', fp,'fn', fn,'tp', tp,
                      'prec',prec,'rec',rec,'fbeta',fbeta,
                      'tpr',tpr,'fpr',fpr,'corr',corr)
                temp_threshold.append([tn, fp, fn, tp])
            temp_band.append(temp_threshold)
        results_in_list[str(sub)+'_'+str(day)]=temp_band
 
    except:
        print('no annotation found')

pickle.dump( validation, open( "validation_fixed_channel.p", "wb" ) )
pickle.dump( validations, open( "validations_fixed_channel.p", "wb" ) )
pickle.dump( results_in_list, open( "results_in_list_fixed_channel.p", "wb" ) )