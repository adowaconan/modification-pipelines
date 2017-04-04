# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:12:36 2017

@author: ning
"""

import numpy as np
import pickle
from random import shuffle
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import os
import eegPinelineDesign
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
os.chdir('D:\\NING - spindle\\training set')
l,h=11,16
folder='step_size_500_%d_%dgetting_higher_threshold\\'%(l,h)
# find the best threshold after we sampled the data
print('find best threshold')
"""
with open('over_threshold.p','rb') as h:
    over_threshold = pickle.load(h)
type_ = 'with'
df_with = eegPinelineDesign.compute_measures(over_threshold,type_,plot_flag=True)
threshold = df_with['thresholds'][np.argmax(df_with['AUC'].mean(1))]  
"""

with open('%sover_threshold.p'%folder,'rb') as h:
    over_threshold = pickle.load(h)
type_ = 'with'
df_accuracy,df_confusion_matrix,df_fpr,df_tpr,df_AUC,threshold_list,df_with = eegPinelineDesign.compute_two_thresholds(over_threshold,type_,plot_flag=False)
best = df_with[df_with['mean_AUC']>round(df_with['mean_AUC'].max(),2)]
best = best[best['mean_accuracy'] == best['mean_accuracy'].max()]
lower_threshold,higher_threshold=best['lower_threshold'].values[0],best['upper_threshold'].values[0]
print('find best threshold: %.2f, %.2f'%(lower_threshold,higher_threshold))  
### train the exported pipeline ###
print('get data')
samples = pickle.load(open("%s%.2f-%.2f_samples_%s.p"%(folder,lower_threshold,higher_threshold,type_),
                                "rb"))
label = pickle.load(open("%s%.2f-%.2f_labels_%s.p" %(folder,lower_threshold,higher_threshold,type_),
                        "rb"))
# set up for naive machine learning
#print('set up naive machine learning model')
#data,label,idx_row = np.array(samples),np.array(label),np.arange(0,len(label),1)
#for ii in range(100):
#    shuffle(idx_row)

#data,label = data[idx_row,:],label[idx_row]
#features = data
#tpot_data=pd.DataFrame({'class':label},columns=['class'])
#training_features, testing_features, training_classes, testing_classes = \
#    train_test_split(features, tpot_data['class'], random_state=42)
data,label,idx_row = np.concatenate(samples),np.concatenate(label),np.arange(0,len(label),1)
print('shuffle')
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.80)
print('model selection')
tpot = TPOTClassifier(generations=10, population_size=25,
                      verbosity=2,random_state=373849,num_cv_folds=5,scoring='roc_auc' )
tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)
tpot.export('%s%s_tpot_exported_pipeline.py'%(folder,type_) )  
print('finished model selection')
"""
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import KFold
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(lambda X: X),
        FunctionTransformer(lambda X: X)
    ),
    SelectFwe(alpha=0.05, score_func=f_classif),
    KNeighborsClassifier(n_neighbors=5, weights="distance")
)
fig,ax = plt.subplots(figsize=(16,16))
kf = KFold(n_splits=10,random_state=123,shuffle=True)
results,auc=[],[]
fpr,tpr={},{}
cnt = 0
for train_index, test_index in kf.split(features):
    training_features, testing_features = features[train_index],features[test_index]
    training_classes, testing_classes = tpot_data['class'].values[train_index],tpot_data['class'].values[test_index]

    exported_pipeline.fit(training_features, training_classes)
    results.append(exported_pipeline.predict(testing_features))
    fpr[cnt], tpr[cnt], thresholds = metrics.roc_curve(testing_classes,np.array(results[cnt]).reshape(-1,))
    #fp.append(fpr);tp.append(tpr)
    auc.append(metrics.roc_auc_score(testing_classes,np.array(results[cnt]).reshape(-1,)))
    cnt += 1
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(auc))]))
mean_tpr = np.zeros_like(all_fpr)
from scipy import interp
for i in range(len(auc)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
mean_tpr /= 10
auc = np.array(auc)
ax.plot(all_fpr,mean_tpr,label='%s sleep stage information,\nAverage area under the curve (10 cv): %.3f'%(type_,auc.mean()))
    
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set(xlabel='false alarm rate (classify a "Miss" as a "False alarm spindle")',ylabel='true positive rate',
       title='best model to seperate false alarms and misses in the threshold detection results',
       xlim=[0.,1.],ylim=[0.,1.05])
ax.annotate('best threshold: %.2f'%threshold,xy=(0.1,1))
ax.legend()
fig.savefig('KNN.png')
"""        
    
    
