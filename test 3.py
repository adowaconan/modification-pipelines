# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:55:46 2017

@author: ning
"""

import numpy as np
import mne
from sklearn.model_selection import StratifiedKFold,permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from Filter_based_and_thresholding import Filter_based_and_thresholding
import pandas as pd
import os
os.chdir('D:\\NING - spindle\\training set')
raw_file = 'suj8_d2_nap.fif'
a_file = 'suj8_d2final_annotations.txt'
annotations = pd.read_csv(a_file)        
raw = mne.io.read_raw_fif(raw_file,)
a=Filter_based_and_thresholding()
a.get_raw(raw)
a.get_epochs()
a.get_annotation(annotations)
a.mauanl_label()
epochs = a.epochs
labels = a.manual_labels
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=12345)
clf = make_pipeline(StandardScaler(),SVC(class_weight='balanced',random_state=12345))
td = mne.decoding.TimeDecoding(cv=cv,clf=clf,scorer='roc_auc',times={'step':0.05,'length':0.05},n_jobs=4)
td.fit(epochs,labels,)
td.score(epochs,labels)
td.plot()
data = epochs.get_data()[:,:,:-1]
chunk = np.array(list(zip(np.arange(0,3.05,0.05)[:-1],np.arange(0,3.05,0.05)[1:])))
results = {'scores':[],'sig':[]}
for slices in (chunk* epochs.info['sfreq']).astype(int):
    temp_data = data[:,:,slices[0]:slices[1]]
    temp_data = mne.decoding.Vectorizer().fit_transform(temp_data)
    score,_,pValue = permutation_test_score(clf,temp_data,labels,cv=cv,random_state=12345,scoring='roc_auc',n_jobs=4)
    results['scores'].append(score)
    results['sig'].append(pValue)

