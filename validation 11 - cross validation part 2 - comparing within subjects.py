# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:22:12 2017

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
exported_pipeline = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.56, max_features=0.56, n_estimators=500,random_state=1093)
)  
kf = KFold(n_splits=10,random_state=556,shuffle=True)
results,auc=[],[]
cnt = 0
print('machine learning model best ML model and cross validation by 10 folds') 
fp,tp=[],[]   
for train_index, test_index in kf.split(features):
    training_features, testing_features = features[train_index],features[test_index]
    training_classes, testing_classes = tpot_data['class'].values[train_index],tpot_data['class'].values[test_index]

    exported_pipeline.fit(training_features, training_classes)
    results.append(exported_pipeline.predict(testing_features))
    fpr, tpr, thresholds = metrics.roc_curve(testing_classes,np.array(results[cnt]).reshape(-1,))
    auc.append(metrics.roc_auc_score(testing_classes,np.array(results[cnt]).reshape(-1,)))
    #ax.plot(fpr,tpr,label='%s,Area under the curve: %.3f'%(type_,auc[cnt]))
    fp.append(fpr);tp.append(tpr)
    cnt += 1
fpr = np.mean(fp,0);tpr=np.mean(tp,0)
fig,ax = plt.subplots(figsize=(16,16))
ax.plot(fpr,tpr,label='Area under the curve: %.3f'%(np.mean(auc)))
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set(xlabel='false alarm rate',ylabel='true positive rate',
       xlim=[0.,1.],ylim=[0.,1.05])
ax.set_title('Gradient Boosting Classifier model to classify spindles and non spindles',fontsize=22,fontweight="bold")
ax.annotate('"GradientBoostingClassifier": learning_rate=0.56,\n               max_features=0.56,\n               n_estimators=500\n               "K-fold": 10, shuffle',xy=(0.04,0.85))
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
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

all_predictions_KNN,all_detections,all_predictions_randomforest={},{},{}
running_time_KNN,running_time_signal_process,running_time_randomforest={},{},{}

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
        # machine learning model KNN
        print('KNN model for %s' % (sub+day))
        t0=time.time()
        all_predictions_KNN[for_name]=eegPinelineDesign.fit_data(raw,KNN,
                                          annotation_file,cv)
        t1=time.time()
        running_time_KNN[for_name]=t1-t0
        print('KKN cv time: %.2f s'%(t1-t0))

        # machine learning model randomforest
        print('ranomd forest model for %s' % (sub+day))
        t0=time.time()
        all_predictions_randomforest[for_name]=eegPinelineDesign.fit_data(raw,randomforest,
                                          annotation_file,cv)
        t1=time.time()
        running_time_randomforest[sub+day]=t1-t0
        print('RF cv time: %.2f s'%(t1-t0))
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
        
pickle.dump([all_detections,all_predictions_KNN,all_predictions_randomforest],open("%smodel comparions.p"%folder,"wb"))
pickle.dump([running_time_signal_process,running_time_KNN,running_time_randomforest],open("%smodel running time.p"%folder,"wb"))
result = pickle.load(open("%smodel comparions.p"%folder,"rb"))
all_detections,all_predictions_KNN,all_predictions_randomforest = result

############################################################################
fig, ax = plt.subplots(figsize=(16,16));cnt = 0
xx,yy,xerr,ylabel = [],[],[],[]
for keys, item in all_predictions_KNN.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
            label='individual performance')

ax.axvline(xx.mean(),
           label='average performance across subjects: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
sortylabel = [ylabel[ii] for ii in sortIdx ]
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+0.5))
ax.annotate('Pipeline: "feature selection": f_classifier\n               "KNN": 5 neighbors, 20 leaves, \n                           metric: minkowski, \n                           weights: distance\n               "K-fold": 10, shuffle',xy=(0.6,13))
ax.set_title('Individual cross validation results by KNN:\ncv=10',fontsize=20,fontweight='bold')
_=ax.legend(loc='best')
fig.savefig('%sindividual performance machine learning_KNN.png'%folder)

fig, ax = plt.subplots(figsize=(16,16));cnt = 0
xx,yy,xerr,ylabel = [],[],[],[]
for keys, item in all_detections.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
            label='individual performance')

ax.axvline(xx.mean(),
           label='average performance across subjects: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+0.5))
ax.set_title('Individual cross validation results by signal processing:\ncv=10',fontsize=20,fontweight='bold')
_=ax.legend(loc='best')
fig.savefig('%sindividual performance signal processing compared to KNN.png'%folder)
######################################################################################


######################################################################################
#fig, ax = plt.subplots(figsize=(16,16));cnt = 0
#xx,yy,xerr,ylabel = [],[],[],[]
#for keys, item in all_predictions_randomforest.items():
#    yy.append(cnt)
#    xx.append(np.mean(item))
#    xerr.append(np.std(item))
#    ylabel.append(keys)
#    cnt += 1
#xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
#sortIdx = np.argsort(xx)
#ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
#            label='individual performance')
#sortylabel = [ylabel[ii] for ii in sortIdx ]
#ax.axvline(xx.mean(),
#           label='average performance across subjects: %.3f'%xx.mean())
#ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
#           xx.mean()+xx.std()/np.sqrt(len(xx)),
#            alpha=0.3,color='blue')
#_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=ylabel,
#        xlabel='Area under the curve on predicting spindles and non spindles',
#        ylabel='Subjects',
#         title='Individual cross validation results by random forest:\ncv=10',
#        ylim=(-0.5,len(ylabel)+0.5))
#_=ax.legend(loc='best')
#fig.savefig('%sindividual performance machine learning_randomforest.png'%folder)

fig, ax = plt.subplots(figsize=(16,16));cnt = 0
xx,yy,xerr,ylabel = [],[],[],[]
for keys, item in all_detections.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
            label='individual performance')

ax.axvline(xx.mean(),
           label='average performance across subjects: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
sortylabel = [ylabel[ii] for ii in sortIdx ]
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+0.5))
ax.set_title('Individual cross validation results by thresholding:\ncv=10',fontsize=20,fontweight='bold')
_=ax.legend(loc='best')
_=ax.annotate('best lower threshold: %.2f, \nbest higher threshold: %.2f' % (lower_threshold,higher_threshold),xy=(0.47,cnt-8))
fig.savefig('%sindividual performance signal processing (itself).png'%folder)
###################################################################################

# put them together
fig, ax = plt.subplots(figsize=(16,16));cnt = 0
xx,yy,xerr,ylabel = [],[],[],[]
for keys, item in all_detections.items():
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='blue',
            label='individual performance_thresholding')
ax.axvline(xx.mean(),color='blue',
           label='thresholding performance: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
sortylabel = [ylabel[ii] for ii in sortIdx ]
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+0.5),
        xlim=(0.45,1.))
ax.set_title('Individual model comparison results:\ncv=10',fontsize=20,fontweight='bold')
#xx,yy,xerr,ylabel = [],[],[],[];cnt=0
#for keys, item in all_predictions_randomforest.items():
#    yy.append(cnt+0.1)
#    xx.append(np.mean(item))
#    xerr.append(np.std(item))
#    ylabel.append(keys)
#    cnt += 1
#xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
##sortIdx = np.argsort(xx)
#ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='black',
#            label='individual performance_randomforest')
#ax.axvline(xx.mean(),color='black',
#           label='randomforest performance: %.3f'%xx.mean())
#ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
#           xx.mean()+xx.std()/np.sqrt(len(xx)),
#            alpha=0.3,color='black')
from mpl_toolkits.axes_grid.inset_locator import inset_axes
xx,yy,xerr,ylabel = [],[],[],[];cnt = 0
for keys, item in all_predictions_KNN.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='red',
            label='individual performance_KNN')

ax.axvline(xx.mean(),
           label='KNN performance: %.3f'%xx.mean(),color='red')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='red')

#ldg=ax.legend(bbox_to_anchor=(1.5, 0.8))
ldg =ax.legend(loc='center right',prop={'size':18})

file = 'suj13_l2nap_day2.fif'
annotation_file = ['suj13_nap_day2_edited_annotations.txt']
raw = mne.io.read_raw_fif(file,preload=True)
annotation = pd.read_csv(annotation_file[0])
raw.resample(500, npad="auto")
raw.pick_channels(channelList)
raw.filter(low,high)
stop = raw.times[-1]-300
e = mne.make_fixed_length_events(raw,1,start=100,stop = stop,duration=3,)
manual_label = eegPinelineDesign.discritized_onset_label_manual(raw,annotation,3)
epochs = mne.Epochs(raw,e,1,tmin=0,tmax=3,proj=False,preload=True)
psds, freqs=mne.time_frequency.psd_multitaper(epochs,tmin=0,tmax=3,low_bias=True,proj=False,)
psds = 10* np.log10(psds)
data = epochs.get_data()[:,:,:-1];freqs = freqs[psds.argmax(2)];psds = psds.max(2); 
freqs = freqs.reshape(len(freqs),6,1);psds = psds.reshape(len(psds),6,1)
data = np.concatenate([data,psds,freqs],axis=2)
data = data.reshape(len(e),-1)

predictions = [];fp,tp=[],[];auc=[];cnt=0
for train, test in cv.split(manual_label):
    
    #KNN.fit(data[train],label[train])
    predictions.append(KNN.predict(data[test]))
    fpr, tpr, thresholds = metrics.roc_curve(manual_label[test],np.array(predictions[cnt]).reshape(-1,))
    auc.append(metrics.roc_auc_score(manual_label[test],np.array(predictions[cnt]).reshape(-1,)))
    #ax.plot(fpr,tpr,label='%s,Area under the curve: %.3f'%(type_,auc[cnt]))
    fp.append(fpr);tp.append(tpr)
    cnt += 1
fpr = np.mean(fp,0);tpr=np.mean(tp,0)
_,fpr,tpr,_ = eegPinelineDesign.fit_data(raw,KNN,annotation_file,cv=cv,plot_flag=True)
inset_axes1 = inset_axes(ax, 
                    width="40%", 
                    height=3.0, 
                    loc=1)

_,fpr1,tpr1 = eegPinelineDesign.fit_data(raw,KNN,annotation_file,cv,plot_flag=True)
inset_axes2 = inset_axes(ax, 
                    width="40%", 
                    height=3.0, 
                    loc=4)
fig.tight_layout()
fig.savefig('%sindividual performance (2 models).png'%folder,bbox_extra_artists=(ldg,), bbox_inches='tight')
