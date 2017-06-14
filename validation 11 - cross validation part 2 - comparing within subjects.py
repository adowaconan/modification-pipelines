# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:22:12 2017
Update on Mon April 10 15:25:26 2017
Updata on Mon June 12 11:42:33 2017
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


print('model comparison')

#exported_pipeline = joblib.load('%smy_model.pkl'%folder)
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
kf = KFold(n_splits=10,random_state=556,shuffle=True)
results,auc=[],[]
"""
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
legend=ax.legend(loc='upper left',frameon=False)
frame = legend.get_frame()
frame.set_facecolor('None')
fig.savefig('%sGBC.png'%folder)
"""
# individual cross validation
exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        )
fitting_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        )
cv = KFold(n_splits=5,random_state=18375,shuffle=True) #maybe we should reduce the number of splits to 5

all_predictions_ML,all_detections,all_predictions_randomforest={},{},{}
running_time_ML,running_time_signal_process,running_time_randomforest={},{},{}
for file in list_file_to_read:
    if file == 'suj20_l2nap_day2.fif':
        pass
    else:
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
            if (sub != 'suj23'):# and (sub != 'suj19'):
                all_predictions_ML[for_name]=eegPinelineDesign.fit_data(raw,exported_pipeline,
                                              annotation_file,cv=cv)
            else:
                print('alternative')
                fitting_pipeline.fit(data,label)
                all_predictions_ML[for_name]=eegPinelineDesign.fit_data(raw,fitting_pipeline,
                                              annotation_file,cv=cv,few=True)
            
            t1=time.time()
            running_time_ML[for_name]=t1-t0
            print('export pipeline cv time: %.2f s, auc: %.2f'%(t1-t0,np.mean(all_predictions_ML[for_name][0])))
    
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
                                                                                        l,h,annotation_file,cv=cv)
            t1=time.time()
            running_time_signal_process[for_name]=t1-t0
            print('my model cv time: %.2f s, auc: %.2f'%(t1-t0,np.mean(all_detections[for_name][0])))
            
                
        else:
            print(sub+day+'no annotation')
        
pickle.dump([all_detections,all_predictions_ML,all_predictions_randomforest],open("%smodel comparions.p"%folder,"wb"))
pickle.dump([running_time_signal_process,running_time_ML,running_time_randomforest],open("%smodel running time.p"%folder,"wb"))

"""
result = pickle.load(open("%smodel comparions.p"%folder,"rb"))
df = {'ML':[],'threshold':[]}
all_detections,all_predictions_ML,_ = result 
for (key1, ML),(key2, detect) in zip(all_predictions_ML.items(),all_detections.items()):
    df['ML'].append(ML[0])
    df['threshold'].append(detect[0])
    
df['ML']=np.concatenate(df['ML'])
df['threshold']=np.concatenate(df['threshold'])


exported_pipeline = make_pipeline(
    make_union(VotingClassifier([("est", DecisionTreeClassifier())]), FunctionTransformer(lambda X: X)),
    GradientBoostingClassifier(learning_rate=0.24, max_features=0.24, n_estimators=500)
        ) 
############################################################################
fig, axes = plt.subplots(figsize=(16,32),nrows=2,sharex=True);cnt = 0
ax=axes[0]
xx,yy,xerr,ylabel = [],[],[],[]
for keys, (item,fpr,tpr) in all_predictions_ML.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
            label='individual performance')

ax.axvline(xx.mean(),
           label='average performance across subjects: %.3f'%xx.mean(),color='black')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
sortylabel = [ylabel[ii] for ii in sortIdx ]
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+0.5),
        )
ax.annotate(exported_pipeline.named_steps,xy=(0.2,26),size=8)
#ax.annotate(exported_pipeline.named_steps,xy=(0.03,0.77),size=8)
ax.set_title('Individual cross validation results by KNN:\ncv=10',fontsize=20,fontweight='bold')
lgd=ax.legend(loc='lower right',frameon=False,prop={'size':18})
frame = lgd.get_frame()
frame.set_facecolor('None')
#fig.savefig('%sindividual performance machine learning_ML.png'%folder)

#fig, ax = plt.subplots(figsize=(16,16));
cnt = 0;ax=axes[1]
xx,yy,xerr,ylabel = [],[],[],[]
for keys, (item,fpr,tpr) in all_detections.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',
            label='individual performance')
sortylabel = [ylabel[ii] for ii in sortIdx ]
ax.axvline(xx.mean(),
           label='average performance across subjects: %.3f'%xx.mean())
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),
            alpha=0.3,color='blue')
_=ax.set(yticks = np.arange(len(ylabel)),yticklabels=sortylabel,
        xlabel='Area under the curve on predicting spindles and non spindles',
        ylabel='Subjects',
        ylim=(-0.5,len(ylabel)+0.5),
        )
ax.set_title('Individual cross validation results by signal processing:\ncv=10',fontsize=20,fontweight='bold')
lgd=ax.legend(loc='lower right',frameon=False,prop={'size':18})
frame = lgd.get_frame()
frame.set_facecolor('None')
fig.savefig('%ssubplots'%folder)
#fig.savefig('%sindividual performance signal processing compared to KNN.png'%folder)
fig.savefig('%sindividual performance subplots.png'%folder)
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

fig, ax = plt.subplots(figsize=(20,16));cnt = 0
xx,yy,xerr,ylabel = [],[],[],[]
for keys, (item,fpr,tpr) in all_detections.items():
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
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
lgd=ax.legend(loc='lower right',frameon=False,prop={'size':18})
frame = lgd.get_frame()
frame.set_facecolor('None')
_=ax.annotate('best lower threshold: %.2f, \nbest higher threshold: %.2f' % (lower_threshold,higher_threshold),xy=(0.47,cnt-8))
fig.savefig('%sindividual performance signal processing (itself).png'%folder)
"""
###################################################################################
#from sklearn.externals import joblib
#exported_pipeline = joblib.load('%smy_model.pkl'%folder)
#cv = KFold(n_splits=10,random_state=18374,shuffle=True)
# put them together
import seaborn as sns
sns.set_style('white')
fig= plt.figure(figsize=(24,16));cnt = 0;uv=4.7
ax = fig.add_subplot(131)
xx,yy,xerr,ylabel,kk = [],[],[],[],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    kk.append(keys)
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='blue',
            label='FBT, individual')
ax.axvline(xx.mean(),color='blue',ymax=len(ylabel)/(len(ylabel)+uv),
           )
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='blue',label='FBT average: %.3f $\pm$ %.3f'%(xx.mean(),xx.std()))
sortylabel = [ylabel[ii] for ii in sortIdx ]
_=ax.set(ylim=(-0.5,len(ylabel)+4),)
ax.set_ylabel('Subjects',fontsize=16)
plt.yticks(np.arange(len(ylabel)),sortylabel,fontsize=16)
ax.set_title('Model comparison:\ncv=10',fontsize=20,fontweight='bold')
ax.set_xlabel('AUC scores',fontsize=17)

xx,yy,xerr,ylabel,k = [],[],[],[],[];cnt = 0
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    k.append(keys)
    yy.append(cnt)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)

ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='red',
            label='Machine learning, individual')

ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           color='red')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='red',label='Machine learning average: %.3f $\pm$ %.3f'%(xx.mean(),xx.std()),)
lgd =ax.legend(loc='upper left',prop={'size':12},frameon=False)
frame = lgd.get_frame()
frame.set_facecolor('None')

ax_ML = fig.add_subplot(232)
AUC,fpr,tpr,confM,sensitivity,specificity = all_predictions_ML['suj22day1']
select = np.random.choice(np.arange(5),size=1)[0]
fpr = fpr[select];tpr = tpr[select]
ax_ML.plot(fpr,tpr,label='AUC score: %.3f $\pm$ %.4f'%(np.mean(AUC),np.std(AUC)),color='red')
ax_ML.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
l=ax_ML.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_ML.set_title('subject %d day %d\nMachine learning model'%(22,1),fontweight='bold',fontsize=20)
ax_ML.set(ylim=(0,1.02),xlim=(0,1.02))
ax_ML.set_ylabel('True positive rate',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_signal = fig.add_subplot(235)
temp_auc,fp,tp,confM,sensitivity,specificity = all_detections['suj22day1']
fp,tp = np.array(fp),np.array(tp)
ax_signal.plot(fp,tp,label='AUC score: %.3f $\pm$ %.4f'%(np.mean(temp_auc),np.std(temp_auc)),color='blue')
ax_signal.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_signal.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_signal.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal.set(ylim=(0,1.02),xlim=(0,1.02))
ax_signal.set_ylabel('True positive rate',fontsize=16)
ax_signal.set_xlabel('False positive rate',fontsize=17)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_ML_CM = fig.add_subplot(233)
AUC,fpr,tpr,confM,sensitivity,specificity = all_predictions_ML['suj22day1']
select = np.random.choice(np.arange(5),size=1)[0]
fpr = fpr[select];tpr = tpr[select]
ax_ML_CM=sns.heatmap(np.mean(confM,0).reshape(2,2),cbar=False,cmap=plt.cm.Blues,vmin=0,vmax=1.,
               ax=ax_ML_CM,annot=True)
ax_ML_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_ML_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_ML_CM.set_title('subject %d day %d, confusion matrix\nMachine learning model'%(22,1),fontweight='bold',fontsize=20)
ax_ML_CM.set_ylabel('True label',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_signal_CM = fig.add_subplot(236)
temp_auc,fp,tp,confM,sensitivity,specificity = all_detections['suj22day1']
fp,tp = np.array(fp),np.array(tp)
ax_signal_CM=sns.heatmap(np.mean(confM,0).reshape(2,2),cbar=False,cmap=plt.cm.Blues,vmin=0,vmax=1.,
               ax=ax_signal_CM,annot=True)
ax_signal_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_signal_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_signal_CM.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal_CM.set_ylabel('True label',fontsize=16,)
ax_signal_CM.set_xlabel('Predicted label',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


#fig.tight_layout()
fig.savefig('%sindividual performance (2 models) (edited).png'%folder, bbox_inches='tight')


#################################################################################
fig,axes = plt.subplots(nr)




































