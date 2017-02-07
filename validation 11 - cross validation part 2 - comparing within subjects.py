# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:22:12 2017

@author: install
"""

import numpy as np
import pandas as pd
import pickle
from random import shuffle
from tpot import TPOTClassifier
import mne
import eegPinelineDesign
from sklearn.model_selection import train_test_split
from sklearn import metrics


file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]

l,h = (11,16);all_predictions={}
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
        raw = mne.io.read_raw_fif(file,preload=True)
        if old:
            pass
        else:
            raw.resample(500, npad="auto")
        raw.pick_channels(channelList)
        raw.filter(l,h)
        
        initial_time = 100;
        predictions=eegPinelineDesign.fit_data(raw,tpot)
        annotation = pd.read_csv(annotation_file[0])
        gold_standard = eegPinelineDesign.read_annotation(raw,annotation_file)
        manual_labels = eegPinelineDesign.discritized_onset_label_manual(raw,gold_standard,3)
               
    else:
        print(sub+day+'no annotation')