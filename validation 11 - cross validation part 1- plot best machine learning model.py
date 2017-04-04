# -*- coding: utf-8 -*-
"""
Created on Tue april  1 13:22:12 2017

@author: ning
"""

import numpy as np
import pandas as pd
import pickle
from random import shuffle
import mne
import eegPinelineDesign
#from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':22})
matplotlib.rcParams['legend.numpoints'] = 1

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFwe, f_classif
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

import time

## standard setup
print('setup')
file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=500;syn_channel=3

# find the best threshold after we sampled the data

low = 11; high = 16
l,h=11,16
folder='step_size_500_%d_%dgetting_higher_threshold\\'%(l,h)
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
print('set up naive machine learning model')
data,label = np.concatenate(samples),np.concatenate(label)
idx_row=np.arange(0,len(label),1)
print('shuffle')
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
features = data
tpot_data=pd.DataFrame({'class':label},columns=['class'])

# train the machine learning model  
 
kf = KFold(n_splits=10,random_state=556,shuffle=True)
results,auc=[],[]
cnt = 0
print('machine learning model best ML model and cross validation by 10 folds') 
fp,tp=[],[]   
for train_index, test_index in kf.split(features):
    training_features, testing_features = features[train_index],features[test_index]
    training_classes, testing_classes = tpot_data['class'].values[train_index],tpot_data['class'].values[test_index]
    exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        ) 
    exported_pipeline.fit(training_features, training_classes)
    results.append(exported_pipeline.predict_proba(testing_features)[:,1])
    fpr, tpr, thresholds = metrics.roc_curve(testing_classes,exported_pipeline.predict_proba(testing_features)[:,1])
    auc.append(metrics.roc_auc_score(testing_classes,exported_pipeline.predict_proba(testing_features)[:,1]))
    #ax.plot(fpr,tpr,label='%s,Area under the curve: %.3f'%(type_,auc[cnt]))
    fp.append(fpr);tp.append(tpr)
    print('get one done')
    cnt += 1
print('done')
#from sklearn.externals import joblib
#pickle.dump(exported_pipeline, open('%smy_model.pkl'%folder,'wb'))
#exported_pipeline = joblib.load('%smy_model.pkl'%folder)
pickle.dump([results,auc,fp,tp],open("%slong process.p"%folder,"wb"))
dd = pickle.load(open('%slong process.p'%folder,'rb'))
results,auc,fp,tp= dd
select = np.random.choice(np.arange(10),size=1)[0]
fpr = fp[select];tpr=tp[select]
fig,ax = plt.subplots(figsize=(16,16))
ax.plot(fpr,tpr,label='Area under the curve: %.3f $\pm$ %.4f'%(np.mean(auc),np.std(auc)))
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set(xlabel='false alarm rate',ylabel='true positive rate',
       xlim=[0.,1.],ylim=[0.,1.05])
ax.set_title('Gradient Boosting Classifier model to classify spindles and non spindles',fontsize=22,fontweight="bold")
#ax.annotate(exported_pipeline.named_steps,xy=(0.03,0.77),size=8)
legend=ax.legend(loc='upper left',frameon=False)
frame = legend.get_frame()
frame.set_facecolor('None')
fig.savefig('%sGBC.png'%folder)

## KNN
#KNN = make_pipeline(
#    make_union(
#        FunctionTransformer(lambda X: X),
#        FunctionTransformer(lambda X: X)
#    ),
#    SelectFwe(alpha=0.05, score_func=f_classif),
#    KNeighborsClassifier(n_neighbors=5, weights="distance")
#)
#kf = KFold(n_splits=10,random_state=123,shuffle=True)
#results,auc=[],[]
#cnt = 0
#print('machine learning model KNN and cross validation by 10 folds') 
#fp,tp=[],[]   
#for train_index, test_index in kf.split(features):
#    training_features, testing_features = features[train_index],features[test_index]
#    training_classes, testing_classes = tpot_data['class'].values[train_index],tpot_data['class'].values[test_index]
#
#    KNN.fit(training_features, training_classes)
#    results.append(KNN.predict(testing_features))
#    fpr, tpr, thresholds = metrics.roc_curve(testing_classes,np.array(results[cnt]).reshape(-1,))
#    auc.append(metrics.roc_auc_score(testing_classes,np.array(results[cnt]).reshape(-1,)))
#    #ax.plot(fpr,tpr,label='%s,Area under the curve: %.3f'%(type_,auc[cnt]))
#    fp.append(fpr);tp.append(tpr)
#    cnt += 1
#fpr = np.mean(fp,0);tpr=np.mean(tp,0)
#fig,ax = plt.subplots(figsize=(16,16))
#ax.plot(fpr,tpr,label='Area under the curve: %.3f'%(np.mean(auc)))
#ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#ax.set(xlabel='false alarm rate',ylabel='true positive rate',
#       xlim=[0.,1.],ylim=[0.,1.05])
#ax.set_title('KNeightbors classifier model to classify spindles and non spindles',fontsize=22,fontweight="bold")
#ax.annotate('Pipeline: "feature selection": f_classifier\n               "KNN": 5 neighbors, 20 leaves, \n                           metric: minkowski, \n                           weights: distance\n               "K-fold": 10, shuffle',xy=(0.04,0.85))
#legend=ax.legend(loc='upper left',frameon=False)
#frame = legend.get_frame()
#frame.set_facecolor('None')
#fig.savefig('%sKNN.png'%folder)
#
## randomforest
#
#randomforest = make_pipeline(
#        SelectFwe(alpha=0.05, score_func=f_classif),
#        make_union(VotingClassifier([("est", BernoulliNB(alpha=36.0, binarize=0.21, fit_prior=True))]), 
#                   FunctionTransformer(lambda X: X)),
#        RandomForestClassifier(n_estimators=500)
#    )
#results,auc=[],[]
#cnt = 0
#fp,tp=[],[]
#print('machine learning model randomforest and cross validation by 10 folds')    
#for train_index, test_index in kf.split(features):
#    training_features, testing_features = features[train_index],features[test_index]
#    training_classes, testing_classes = tpot_data['class'].values[train_index],tpot_data['class'].values[test_index]
#
#    randomforest.fit(training_features, training_classes)
#    results.append(randomforest.predict(testing_features))
#    fpr, tpr, thresholds = metrics.roc_curve(testing_classes,np.array(results[cnt]).reshape(-1,))
#    auc.append(metrics.roc_auc_score(testing_classes,np.array(results[cnt]).reshape(-1,)))
#    #ax.plot(fpr,tpr,label='%s,Area under the curve: %.3f'%(type_,auc[cnt]))
#    fp.append(fpr);tp.append(tpr)
#    cnt += 1
#fpr = np.mean(fp,0);tpr=np.mean(tp,0)
#fig,ax = plt.subplots(figsize=(16,16))
#ax.plot(fpr,tpr,label='Area under the curve: %.3f'%(np.mean(auc)))
#ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#ax.set(xlabel='false alarm rate',ylabel='true positive rate',
#       xlim=[0.,1.],ylim=[0.,1.05])
#ax.set_title('Rondom forest model to classify spindles and non spindles',fontsize=22,fontweight="bold")
#ax.annotate('Pipeline: "feature selection": f_classifier\n               "Random forest": 500 estimators',xy=(0.04,0.85))
#legend=ax.legend(loc='upper left',frameon=False)
#frame = legend.get_frame()
#frame.set_facecolor('None')
#fig.savefig('%srandomforest.png'%folder)





### applying this machine learning model to our data ###
### also compare to signal processing method in the individual level ###
print('model comparison')
cv = KFold(n_splits=10,random_state=18374,shuffle=True)

all_predictions_ML,all_detections,all_predictions_randomforest={},{},{}
running_time_ML,running_time_signal_process,running_time_randomforest={},{},{}
#exported_pipeline = joblib.load('%smy_model.pkl'%folder)
exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        )
exported_pipeline.fit(data,label)
for file in list_file_to_read:
    sub = file.split('_')[0]
    if int(sub[3:]) >= 11:
        day = file.split('_')[2][:4]
        for_name=sub+day
        old = False
    else:
        day = file.split('_')[1]
        for_name = sub+day[0]+'ay'+day[-1]
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
        raw.filter(low,high)
        # machine learning model export model
        print('export model model for %s' % (sub+day))
        t0=time.time()
        all_predictions_ML[for_name]=eegPinelineDesign.fit_data(raw,exported_pipeline,
                                          annotation_file,cv)
        t1=time.time()
        running_time_ML[for_name]=t1-t0
        print('export pipeline cv time: %.2f s'%(t1-t0))

        # machine learning model randomforest
        """
        print('ranomd forest model for %s' % (sub+day))
        t0=time.time()
        all_predictions_randomforest[for_name]=eegPinelineDesign.fit_data(raw,randomforest,
                                          annotation_file,cv)
        t1=time.time()
        running_time_randomforest[sub+day]=t1-t0
        print('RF cv time: %.2f s'%(t1-t0))
        """
        
        # signal processing model
        print('signal processing model for %s' % (sub+day))
        t0=time.time()
        all_detections[for_name]=eegPinelineDesign.detection_pipeline_crossvalidation(raw,channelList,
                                                                                    annotation,windowSize,
                                                                                    lower_threshold,higher_threshold,syn_channel,
                                                                                    l,h,annotation_file)
        t1=time.time()
        running_time_signal_process[for_name]=t1-t0
        print('my model cv time: %.2f s'%(t1-t0))
        
            
    else:
        print(sub+day+'no annotation')
        
pickle.dump([all_detections,all_predictions_ML,all_predictions_randomforest],open("%smodel comparions.p"%folder,"wb"))
pickle.dump([running_time_signal_process,running_time_ML,running_time_randomforest],open("%smodel running time.p"%folder,"wb"))
