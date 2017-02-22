# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:22:12 2017

@author: install
"""

import numpy as np
import pandas as pd
import pickle
from random import shuffle
import mne
import eegPinelineDesign
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

import time

## standard setup
print('setup')
file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=500;threshold=0.85;syn_channel=3

# find the best threshold after we sampled the data
print('find best threshold')
with open('over_threshold.p','rb') as h:
    over_threshold = pickle.load(h)
type_ = 'with'
df_with = eegPinelineDesign.compute_measures(over_threshold,type_,plot_flag=True)
threshold = df_with['thresholds'][np.argmax(df_with['AUC'].mean(1))]                    
### train the exported pipeline ###
samples = pickle.load(open("temp_data\\%.2f_samples_%s.p"%(threshold,type_),
                                "rb"))
label = pickle.load(open("temp_data\\%.2f_labels_%s.p" %(threshold,type_),
                        "rb"))
# set up for naive machine learning
print('set up naive machine learning model')
data,label,idx_row = np.array(samples),np.array(label),np.arange(0,len(label),1)
for ii in range(100):
    shuffle(idx_row)
fig,ax = plt.subplots(figsize=(16,16))
data,label = data[idx_row,:],label[idx_row]
features = data
tpot_data=pd.DataFrame({'class':label},columns=['class'])
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)
exported_pipeline = make_pipeline(
        SelectFwe(alpha=0.05, score_func=f_classif),
        make_union(VotingClassifier([("est", BernoulliNB(alpha=36.0, binarize=0.21, fit_prior=True))]), 
                   FunctionTransformer(lambda X: X)),
        RandomForestClassifier(n_estimators=500)
    )
# train the machine learning model
print('machine learning model')
exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
fpr, tpr, thresholds = metrics.roc_curve(testing_classes,results)
auc = metrics.roc_auc_score(testing_classes,results)
ax.plot(fpr,tpr,label='%s,Area under the curve: %.3f'%(type_,auc))
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set(xlabel='false alarm rate (classify a "Miss" as a "False alarm spindle")',ylabel='true positive rate',
       title='best model to seperate false alarms and misses in the threshold detection results',
       xlim=[0.,1.],ylim=[0.,1.05])
ax.legend()
#fig.savefig('randomforest.png')

### applying this machine learning model to our data ###
### also compare to signal processing method in the individual level ###
print('model comparison')
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
l,h = (11,16);all_predictions,all_detections={},{}
running_time_machine_learning,running_time_signal_process={},{}

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
        annotation = pd.read_csv(annotation_file[0])
        raw = mne.io.read_raw_fif(file,preload=True)
        if old:
            pass
        else:
            raw.resample(500, npad="auto")
        raw.pick_channels(channelList)
        raw.filter(l,h)
        # machine learning model 
        print('machine learning model for %s' % (sub+day))
        t0=time.time()
        all_predictions[sub+day]=eegPinelineDesign.fit_data(raw,exported_pipeline,
                                          annotation_file,cv)
        t1=time.time()
        running_time_machine_learning[sub+day]=t1-t0
        # signal processing model
        print('signal processing model for %s' % (sub+day))
        t0=time.time()
        all_detections[sub+day]=eegPinelineDesign.detection_pipeline_crossvalidation(raw,channelList,
                                                                                    file,annotation,windowSize,
                                                                                    threshold,syn_channel,
                                                                                    l,h,annotation_file)
        t1=time.time()
        running_time_signal_process[sub+day]=t1-t0
        
            
    else:
        print(sub+day+'no annotation')
        
pickle.dump([all_detections,all_predictions],open("model comparions.p","wb"))
"""
fig, ax = plt.subplots(figsize=(16,16));cnt = 0
xx,yy,xerr,ylabel = [],[],[],[]
for keys, item in all_predictions.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
            label='individual performance in a machine learning model:')

ax.axvline(xx.mean(),
           label='average performance of machine learning model: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=ylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
         title='Individual cross validation results:\ncv=10',
        ylim=(-0.5,len(ylabel)+0.5))
_=ax.legend(loc='best')
#fig.savefig('individual performance.png')
"""