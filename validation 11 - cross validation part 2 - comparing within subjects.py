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
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams['legend.numpoints'] = 1
plt.rc('font',weight='bold',size=16)


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

all_predictions_ML,all_detections,all_detections_alter={},{},{}
running_time_ML,running_time_signal_process,running_time_detections_alter={},{},{}
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
    
            # signal processing with adapted threshold for confusion matrix
            
            print('signal processing with altered threshold for %s' % (sub+day))
            t0=time.time()
            all_detections_alter[for_name]=eegPinelineDesign.detection_pipeline_crossvalidation(raw,channelList,
                                                                                        annotation,windowSize,
                                                                                        lower_threshold,higher_threshold,syn_channel,
                                                                                        l,h,annotation_file,cv=cv,
                                                                                        auc_threshold='adapt')
            t1=time.time()
            running_time_detections_alter[sub+day]=t1-t0
            print('altered my model cv time: %.2f s, auc: %.2f'%(t1-t0,np.mean(all_detections_alter[for_name][0])))
            
            
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
        
pickle.dump([all_detections,all_predictions_ML,all_detections_alter],open("%smodel comparions.p"%folder,"wb"))
pickle.dump([running_time_signal_process,running_time_ML,running_time_detections_alter],open("%smodel running time.p"%folder,"wb"))

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
result = pickle.load(open("%smodel comparions.p"%folder,"rb"))
df = {'ML':[],'threshold':[]}
all_detections,all_predictions_ML,all_detections_alter = result 
import seaborn as sns
import re
from scipy import interp
sns.set_style('white')
####################################################################################
df_all = {'Subject':[],
          'Day':[],
          'Mean TN':[],
          'Mean FP':[],
          'Mean FN':[],
          'Mean TP':[],
          'Mean sensitivity':[],
          'Mean specificity':[],
          'Model':[],
          'Mean AUC':[]}
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    sub = int(keys.split('day')[0][3:])
    day = int(keys.split('day')[1])
    mean_confM = np.mean(confM,0)
    std_confM = np.std(confM,0)
    TN,FP,FN,TP = mean_confM
    TN_,FP_,FN_,TP_ = std_confM
    mean_sensitivity = np.mean(sensitivity)
    std_sensitivity = np.std(sensitivity)
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    mean_AUC = np.mean(item)
    std_AUC = np.std(item)
    df_all['Subject'].append(sub)
    df_all['Day'].append(day)
    df_all['Mean TN'].append(TN)
    df_all['Mean FP'].append(FP)
    df_all['Mean FN'].append(FN)
    df_all['Mean TP'].append(TP)
    df_all['Mean sensitivity'].append(mean_sensitivity)
    df_all['Mean specificity'].append(mean_specificity)
    df_all['Model'].append('Machine learning')
    df_all['Mean AUC'].append(mean_AUC)
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    sub = int(keys.split('day')[0][3:])
    day = int(keys.split('day')[1])
    mean_confM = np.mean(confM,0)
    std_confM = np.std(confM,0)
    TN,FP,FN,TP = mean_confM
    TN_,FP_,FN_,TP_ = std_confM
    mean_sensitivity = np.mean(sensitivity)
    std_sensitivity = np.std(sensitivity)
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    mean_AUC = np.mean(item)
    std_AUC = np.std(item)
    df_all['Subject'].append(sub)
    df_all['Day'].append(day)
    df_all['Mean TN'].append(TN)
    df_all['Mean FP'].append(FP)
    df_all['Mean FN'].append(FN)
    df_all['Mean TP'].append(TP)
    df_all['Mean sensitivity'].append(mean_sensitivity)
    df_all['Mean specificity'].append(mean_specificity)
    df_all['Model'].append('FBT')
    df_all['Mean AUC'].append(mean_AUC) 
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections_alter.items():
    sub = int(keys.split('day')[0][3:])
    day = int(keys.split('day')[1])
    mean_confM = np.mean(confM,0)
    std_confM = np.std(confM,0)
    TN,FP,FN,TP = mean_confM
    TN_,FP_,FN_,TP_ = std_confM
    mean_sensitivity = np.mean(sensitivity)
    std_sensitivity = np.std(sensitivity)
    mean_specificity = np.mean(specificity)
    std_specificity = np.std(specificity)
    mean_AUC = np.mean(item)
    std_AUC = np.std(item)
    df_all['Subject'].append(sub)
    df_all['Day'].append(day)
    df_all['Mean TN'].append(TN)
    df_all['Mean FP'].append(FP)
    df_all['Mean FN'].append(FN)
    df_all['Mean TP'].append(TP)
    df_all['Mean sensitivity'].append(mean_sensitivity)
    df_all['Mean specificity'].append(mean_specificity)
    df_all['Model'].append('FBT_alter')
    df_all['Mean AUC'].append(mean_AUC)
df_all = pd.DataFrame(df_all)   
df_all = df_all.sort_values(['Subject','Day'])
df_all = df_all[['Subject','Day','Mean AUC','Mean TN','Mean FP','Mean FN','Mean TP',
                 'Mean sensitivity','Mean specificity','Model']]
df_all.to_csv('%smore measures.csv'%folder,index=False)
df_all = pd.read_csv('%smore measures.csv'%folder)
df_all = df_all#[df_all['Model'] != 'FBT_alter']
df_all_plot = df_all[['Mean AUC','Mean TN','Mean FP','Mean FN','Mean TP',
                 'Mean sensitivity','Mean specificity','Model']]
df_all_plot.columns = ['AUC','True negative rate','False positive rate','False negative rate','True positive rate',
                 'Sensitivity','Specificity','Model']
df_all_plot['Model'] = df_all_plot['Model'].map({'Machine learning':'Machine learning',
           'FBT':'Filter based\nand thresholding',
           'FBT_alter':'Filter based\nand thresholding\nchanged thresholds'})

g = sns.PairGrid(data=df_all_plot,hue='Model',)
g.map_lower(sns.regplot,)
g.map_upper(plt.scatter, edgecolor="w")
g.map_diag(sns.kdeplot,lw=3)
g.add_legend(title='',fontsize=20)
replacements = df_all_plot.columns
for i in range(7):
    for j in range(7):
        xlabel = g.axes[i][j].get_xlabel()
        ylabel = g.axes[i][j].get_ylabel()
        if xlabel in replacements:
            g.axes[i][j].set_xlabel(xlabel,fontsize=15,fontweight='bold')
        if ylabel in replacements:
            g.axes[i][j].set_ylabel(ylabel,fontsize=15,fontweight='bold')


g.fig.savefig('%smany measures(X3).png'%folder, bbox_inches='tight',dpi=400)
############################################################################################################################
df = pd.read_csv('D:\\NING - spindle\\training set\\step_size_500_11_16getting_higher_threshold\\more measures.csv')
df_FBT = df[df['Model'] == 'FBT'].reset_index()
df_FBT['dist'] = abs(df_FBT['Mean AUC'] - df_FBT['Mean AUC'].median())
flashing_key_FBT = [df_FBT['Subject'][np.argmin(df_FBT['dist'])],df_FBT['Day'][np.argmin(df_FBT['dist'])]]
df_ML = df[df['Model'] == 'Machine learning'].reset_index()
df_ML['dist'] = abs(df_ML['Mean AUC'] - df_ML['Mean AUC'].median())
flashing_key_ML = [df_ML['Subject'][np.argmin(df_ML['dist'])],df_ML['Day'][np.argmin(df_ML['dist'])]]
df_FBT_ = df[df['Model'] == 'FBT_alter'].reset_index()
df_FBT_['dist'] = abs(df_FBT_['Mean AUC'] - df_FBT_['Mean AUC'].median())
flashing_key_FBT_ = [df_FBT_['Subject'][np.argmin(df_FBT_['dist'])],df_FBT_['Day'][np.argmin(df_FBT_['dist'])]]
###########################################################
fig= plt.figure(figsize=(24,16));cnt = 0;uv=4.7
ax = fig.add_subplot(131)
xx,yy,xerr,ylabel,kk = [],[],[],[],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    keys = re.findall('\d+',keys)
    keys = keys[0] +'            ' + keys[1] + ' '
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
ax.errorbar(0,yy[-1]+1,0)
ax.axvline(xx.mean(),color='blue',ymax=len(ylabel)/(len(ylabel)+uv),
           )
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='blue',label='FBT average: %.2f $\pm$ %.2f'%(xx.mean(),
                                                                         xx.std()/np.sqrt(len(xx))))
sortylabel = [ylabel[ii] for ii in sortIdx ]
sortylabel.append('Subject    Day')
_=ax.set(ylim=(-0.5,len(ylabel)+uv),)
plt.xticks(fontsize=16,fontweight='bold')
#ax.set_ylabel('Subjects',fontsize=20,fontweight='bold')
plt.yticks(np.arange(len(ylabel)+1),sortylabel,fontsize=16)
ax.set_title('Model comparison:\ncross validation folds: 5',fontsize=20,fontweight='bold')
ax.set_xlabel('AUC scores',fontsize=20,fontweight='bold')
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
            alpha=0.3,color='red',label='Machine learning average: %.2f $\pm$ %.2f'%(xx.mean(),
                                                                                     xx.std()/np.sqrt(len(xx))),)
lgd =ax.legend(loc='upper left',prop={'size':13},frameon=False)
ax.set(xlim=(0.35,1.))
frame = lgd.get_frame()
frame.set_facecolor('None')
################## second column - machine learning ##########################################
ax_ML = fig.add_subplot(232)
confM_ML = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    keys = re.findall('\d+',keys)
    keys = [int(a) for a in keys]
    
    if len(confM) > 5:
        confM = confM[:5]
    #print(keys,len(confM))
    confM_ML.append(confM)
    if keys == flashing_key_ML:
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        
        ax_ML.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %d, day %d\nAUC: %.2f $\pm$ %.2f'%(keys[0],keys[1],
                                                                                         np.mean(item),
                                                                                         np.std(item)/np.sqrt(len(item))),
                   color='black',alpha=1.)
        ax_ML.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        ax_ML.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:    
        for auc_n in range(5):
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_ML.plot(fpr_plot,tpr_plot,color='red',alpha=0.1)
l=ax_ML.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_ML.set_title('Between subject ROC AUC\nMachine learning model',fontweight='bold',fontsize=20)
ax_ML.set(ylim=(0,1.02),xlim=(0,1.02))
ax_ML.set_ylabel('True positive rate',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
########################### second colum below - FBT ##########################
ax_signal = fig.add_subplot(235)
confM_FBT = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    keys = re.findall('\d+',keys)
    keys = [int(a) for a in keys]
    
    if len(confM) > 5:
        confM = confM[:5]
    #print(keys,len(confM))
    confM_FBT.append(confM)
    if keys == flashing_key_FBT:
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        
        ax_signal.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %d, day %d\nAUC: %.2f $\pm$ %.2f'%(keys[0],keys[1],
                                                                                         np.mean(item),
                                                                                         np.std(item)/np.sqrt(len(item))),
                   color='black',alpha=1.)
        ax_signal.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        ax_signal.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:    
        for auc_n in range(5):
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_signal.plot(fpr_plot,tpr_plot,color='blue',alpha=0.1)
ax_signal.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_signal.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal.set(ylim=(0,1.02),xlim=(0,1.02))
ax_signal.set_ylabel('True positive rate',fontsize=20,fontweight='bold')
ax_signal.set_xlabel('False positive rate',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
########################### 3rd col #####################################################
confM_FBT_mean = np.array(confM_FBT).mean(1).mean(0)
confM_ML_mean = np.array(confM_ML).mean(1).mean(0)
confM_FBT_std = np.array(confM_FBT).mean(1).std(0)/np.sqrt(len(np.array(confM_FBT).mean(1)))
confM_ML_std = np.array(confM_ML).mean(1).std(0)/np.sqrt(len(np.array(confM_ML).mean(1)))
ax_ML_CM = fig.add_subplot(233)
ax_ML_CM=sns.heatmap(confM_ML_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_ML_CM,annot=False)
#coors = np.array([[-0.15,0],[1-0.15,0],[-0.15,1],[1-0.15,1]])+ 0.5
coors = np.array([[0,1],[1,1],[0,0],[1,0],])+ 0.5
for ii, (m,s,coor) in enumerate(zip(confM_ML_mean,confM_ML_std,coors)):
    ax_ML_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_ML_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_ML_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_ML_CM.set_title('Between subject confusion matrix\nMachine learning model',fontweight='bold',fontsize=20)
ax_ML_CM.set_ylabel('True label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_signal_CM = fig.add_subplot(236)
ax_signal_CM=sns.heatmap(confM_FBT_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_signal_CM,annot=False)
for ii, (m,s,coor) in enumerate(zip(confM_FBT_mean,confM_FBT_std,coors)):
    ax_signal_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_signal_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_signal_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_signal_CM.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal_CM.set_ylabel('True label',fontsize=20,fontweight='bold')
ax_signal_CM.set_xlabel('Predicted label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#fig.get_axes()[0].annotate('(a)',(0.35,45.01),fontsize=26)
#fig.get_axes()[1].annotate('(b)',(0.05,1.02),fontsize=26)
#fig.get_axes()[3].annotate('(c)',(0.05,2.0),fontsize=26)
#fig.tight_layout()
fig.savefig('%sindividual performance (2 models) (edited).png'%folder, bbox_inches='tight')
fig.savefig('%sindividual performance (2 models) (high).png'%folder, bbox_inches='tight',
            dpi=300)

#################################################################################
import seaborn as sns
sns.set_style('white')
result = pickle.load(open("%smodel comparions.p"%folder,"rb"))
all_detections,all_predictions_ML,_ = result 

plt.close('all')
fig = plt.figure(figsize=(24,28));cnt = 0
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
sortylabel = [ylabel[ii] for ii in sortIdx[::-1] ]
for cnt,keys in enumerate(sortylabel):
    ax = fig.add_subplot(6,7,cnt+1)
    item,fpr,tpr,confM,sensitivity,specificity = all_detections[keys]
    ax=sns.heatmap(np.mean(confM,0).reshape(2,2),cbar=False,cmap=plt.cm.Blues,vmin=0,vmax=1.,
                   ax=ax,annot=True)
    ax.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
    ax.set_yticklabels(['spindle','non spindle'],rotation=90)
    ax.set_title(keys,fontsize=16,fontweight='bold')
    if cnt % 7 ==0:
        ax.set_ylabel('True label',fontweight='bold',fontsize=16)
    if cnt > 34:
        ax.set_xlabel('Predicted label',fontweight='bold',fontsize=16)
fig.suptitle('Confusion matrix\nFilter based and thresholding model',y=1.015,fontweight='bold')
fig.tight_layout()
fig.savefig('%sconfusion matrix individual thresholding.png'%folder, bbox_inches='tight')
fig.savefig('%sconfusion matrix individual thresholding (high).png'%folder, bbox_inches='tight',
            dpi=300)

plt.close('all')
fig = plt.figure(figsize=(24,28));cnt = 0
xx,yy,xerr,ylabel,kk = [],[],[],[],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    kk.append(keys)
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
sortIdx = np.argsort(xx)
sortylabel = [ylabel[ii] for ii in sortIdx[::-1] ]
for cnt,keys in enumerate(sortylabel):
    ax = fig.add_subplot(6,7,cnt+1)
    item,fpr,tpr,confM,sensitivity,specificity = all_predictions_ML[keys]
    ax=sns.heatmap(np.mean(confM,0).reshape(2,2),cbar=False,cmap=plt.cm.Blues,vmin=0,vmax=1.,
                   ax=ax,annot=True)
    ax.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
    ax.set_yticklabels(['spindle','non spindle'],rotation=90)
    ax.set_title(keys,fontsize=16,fontweight='bold')
    if cnt % 7 ==0:
        ax.set_ylabel('True label',fontweight='bold',fontsize=16)
    if cnt > 34:
        ax.set_xlabel('Predicted label',fontweight='bold',fontsize=16)
fig.suptitle('Confusion matrix\nMachine learning model',y=1.015,fontweight='bold')
fig.tight_layout()
fig.savefig('%sconfusion matrix individual machinelearning.png'%folder, bbox_inches='tight')
fig.savefig('%sconfusion matrix individual machinelearning.png'%folder, bbox_inches='tight',
            dpi=300)
###################################################################################################
# old new testing
from random import shuffle
from scipy.stats import percentileofscore
xx,yy,xerr,ylabel,kk = [],[],[],[],[]
old,new = [],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    kk.append(keys)
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    if int(keys.split('d')[0][3:]) < 11:
        old.append(item)
    else:
        new.append(item)
    cnt += 1
old = np.concatenate(old)
new = np.concatenate(new)
mean_difference = old.mean() - new.mean()
ps = []
for tt in range(500):
    diff = []
    vector_d = np.concatenate([old,new])
    for ii in range(500):
        shuffle(vector_d)
        shuffle_old = vector_d[:len(old)]
        shuffle_new = vector_d[len(old):]
        diff.append(shuffle_old.mean() - shuffle_new.mean())
    ps.append(min(percentileofscore(diff,mean_difference)/100,(100-percentileofscore(diff,mean_difference))/100))
print('p values: %.3f +/- %.3f'%(np.mean(ps),np.std(ps)))
#######################################################################################
from random import shuffle
from scipy.stats import percentileofscore
xx,yy,xerr,ylabel,kk = [],[],[],[],[]
old,new = [],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    kk.append(keys)
    yy.append(cnt+0.1)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    if int(keys.split('d')[0][3:]) < 11:
        old.append(item)
    else:
        new.append(item)
    cnt += 1
old = np.concatenate(old)
new = np.concatenate(new)
mean_difference = old.mean() - new.mean()
ps = []
for tt in range(500):
    diff = []
    vector_d = np.concatenate([old,new])
    for ii in range(500):
        shuffle(vector_d)
        shuffle_old = vector_d[:len(old)]
        shuffle_new = vector_d[len(old):]
        diff.append(shuffle_old.mean() - shuffle_new.mean())
    ps.append(min(percentileofscore(diff,mean_difference)/100,(100-percentileofscore(diff,mean_difference))/100))
print('p values: %.3f +/- %.3f'%(np.mean(ps),np.std(ps)))
####################################################################################################################
####################################################################################################################
fig= plt.figure(figsize=(24,16));cnt = 0;uv=7.0
ax = fig.add_subplot(131)
xx,yy,xerr,ylabel,kk = [],[],[],[],[]
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    keys = re.findall('\d+',keys)
    keys = keys[0] +'            ' + keys[1] + ' '
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
ax.errorbar(0,yy[-1]+1,0)
ax.axvline(xx.mean(),color='blue',ymax=len(ylabel)/(len(ylabel)+uv),
           )
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='blue',label='FBT average: %.2f $\pm$ %.2f'%(xx.mean(),
                                                                         xx.std()/np.sqrt(len(xx))))
sortylabel = [ylabel[ii] for ii in sortIdx ]
sortylabel.append('Subject    Day')
_=ax.set(ylim=(-0.5,len(ylabel)+uv),)
plt.xticks(fontsize=16,fontweight='bold')
#ax.set_ylabel('Subjects',fontsize=20,fontweight='bold')
plt.yticks(np.arange(len(ylabel)+1),sortylabel,fontsize=16)
ax.set_title('Model comparison:\ncross validation folds: 5',fontsize=20,fontweight='bold')
ax.set_xlabel('AUC scores',fontsize=20,fontweight='bold')
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
            alpha=0.3,color='red',label='Machine learning average: %.2f $\pm$ %.2f'%(xx.mean(),
                                                                                     xx.std()/np.sqrt(len(xx))),)
xx,yy,xerr,ylabel,kk = [],[],[],[],[];cnt = 0
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections_alter.items():
    keys = re.findall('\d+',keys)
    keys = keys[0] +'            ' + keys[1] + ' '
    kk.append(keys)
    yy.append(cnt+0.2)
    xx.append(np.mean(item))
    xerr.append(np.std(item)/np.sqrt(len(item)))
    ylabel.append(keys)
    cnt += 1
xx,yy,xerr = np.array(xx),np.array(yy),np.array(xerr)
ax.errorbar(xx[sortIdx],yy,xerr=xerr[sortIdx],linestyle='',color='green',
            label='Improved FBT, individual')
ax.axvline(xx.mean(),ymax=len(ylabel)/(len(ylabel)+uv),
           color='green')
ax.axvspan(xx.mean()-xx.std()/np.sqrt(len(xx)),
           xx.mean()+xx.std()/np.sqrt(len(xx)),ymax=len(ylabel)/(len(ylabel)+uv),
            alpha=0.3,color='green',label='Improved FBT average: %.2f $\pm$ %.2f'%(xx.mean(),
                                                                                     xx.std()/np.sqrt(len(xx))),)

lgd =ax.legend(loc='upper left',prop={'size':13},frameon=False)
ax.set(xlim=(0.35,1.))
frame = lgd.get_frame()
frame.set_facecolor('None')
################## second column - machine learning ##########################################
ax_ML = fig.add_subplot(332)
confM_ML = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    keys = re.findall('\d+',keys)
    keys = [int(a) for a in keys]
    
    if len(confM) > 5:
        confM = confM[:5]
    #print(keys,len(confM))
    confM_ML.append(confM)
    if keys == [29,1]:#flashing_key_ML:
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        
        ax_ML.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %d, day %d\nAUC: %.2f $\pm$ %.2f'%(keys[0],keys[1],
                                                                                         np.mean(item),
                                                                                         np.std(item)/np.sqrt(len(item))),
                   color='black',alpha=1.)
        ax_ML.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        ax_ML.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:    
        for auc_n in range(5):
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_ML.plot(fpr_plot,tpr_plot,color='red',alpha=0.1)
l=ax_ML.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_ML.set_title('Between subject ROC AUC\nMachine learning model',fontweight='bold',fontsize=20)
ax_ML.set(ylim=(0,1.02),xlim=(0,1.02))
ax_ML.set_ylabel('True positive rate',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
########################### second colum middle - FBT ##########################
ax_signal = fig.add_subplot(335)
confM_FBT = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    keys = re.findall('\d+',keys)
    keys = [int(a) for a in keys]
    
    if len(confM) > 5:
        confM = confM[:5]
    #print(keys,len(confM))
    confM_FBT.append(confM)
    if keys == flashing_key_FBT:
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        
        ax_signal.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %d, day %d\nAUC: %.2f $\pm$ %.2f'%(keys[0],keys[1],
                                                                                         np.mean(item),
                                                                                         np.std(item)/np.sqrt(len(item))),
                   color='black',alpha=1.)
        ax_signal.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        ax_signal.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:    
        for auc_n in range(5):
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_signal.plot(fpr_plot,tpr_plot,color='blue',alpha=0.1)
ax_signal.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_signal.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal.set(ylim=(0,1.02),xlim=(0,1.02))
ax_signal.set_ylabel('True positive rate',fontsize=20,fontweight='bold')
ax_signal.set_xlabel('')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
########################### second colum below - improved FBT ##########################
ax_signal = fig.add_subplot(338)
confM_FBT_ = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections_alter.items():
    keys = re.findall('\d+',keys)
    keys = [int(a) for a in keys]
    
    if len(confM) > 5:
        confM = confM[:5]
    #print(keys,len(confM))
    confM_FBT_.append(confM)
    if keys == flashing_key_FBT_:
        tprs = []
        base_fpr = np.linspace(0, 1, 101)
        for select in range(5):
            fpr_ = fpr[select];tpr_ = tpr[select]
            tpr_interp = interp(base_fpr, fpr_, tpr_)
            tpr_interp[0] = 0
            tprs.append(tpr_interp)
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
        
        ax_signal.plot(base_fpr, mean_tprs,
                   label='Most median sample\nsubject %d, day %d\nAUC: %.2f $\pm$ %.2f'%(keys[0],keys[1],
                                                                                         np.mean(item),
                                                                                         np.std(item)/np.sqrt(len(item))),
                   color='black',alpha=1.)
        ax_signal.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        ax_signal.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    else:    
        for auc_n in range(5):
            fpr_plot = fpr[auc_n];tpr_plot = tpr[auc_n]
            ax_signal.plot(fpr_plot,tpr_plot,color='green',alpha=0.1)
ax_signal.legend(loc='lower right',frameon=False,prop={'size':16})
frame = l.get_frame()
frame.set_facecolor('None')
ax_signal.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal.set(ylim=(0,1.02),xlim=(0,1.02))
ax_signal.set_ylabel('True positive rate',fontsize=20,fontweight='bold')
ax_signal.set_xlabel('False positive rate',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
########################### 3rd col #####################################################
confM_FBT_mean = np.array(confM_FBT).mean(1).mean(0)
confM_ML_mean = np.array(confM_ML).mean(1).mean(0)
confM_FBT_std = np.array(confM_FBT).mean(1).std(0)/np.sqrt(len(np.array(confM_FBT).mean(1)))
confM_ML_std = np.array(confM_ML).mean(1).std(0)/np.sqrt(len(np.array(confM_ML).mean(1)))
confM_FBT_mean_ = np.array(confM_FBT_).mean(1).mean(0)
confM_FBT_std_ = np.array(confM_FBT_).mean(1).std(0)/np.sqrt(len(np.array(confM_FBT_).mean(1)))

ax_ML_CM = fig.add_subplot(333)
ax_ML_CM=sns.heatmap(confM_ML_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_ML_CM,annot=False)
#coors = np.array([[-0.15,0],[1-0.15,0],[-0.15,1],[1-0.15,1]])+ 0.5
coors = np.array([[0,1],[1,1],[0,0],[1,0],])+ 0.5
for ii, (m,s,coor) in enumerate(zip(confM_ML_mean,confM_ML_std,coors)):
    ax_ML_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_ML_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_ML_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_ML_CM.set_title('Between subject confusion matrix\nMachine learning model',fontweight='bold',fontsize=20)
ax_ML_CM.set_ylabel('True label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_signal_CM = fig.add_subplot(336)
ax_signal_CM=sns.heatmap(confM_FBT_mean.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_signal_CM,annot=False)
for ii, (m,s,coor) in enumerate(zip(confM_FBT_mean,confM_FBT_std,coors)):
    ax_signal_CM.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_signal_CM.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_signal_CM.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_signal_CM.set_title('Filter based and thresholding model',fontweight='bold',fontsize=20)
ax_signal_CM.set_ylabel('True label',fontsize=20,fontweight='bold')
#ax_signal_CM.set_xlabel('Predicted label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

ax_signal_CM_ = fig.add_subplot(339)
ax_signal_CM_=sns.heatmap(confM_FBT_mean_.reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                     vmin=0,vmax=1.,
                     ax=ax_signal_CM_,annot=False)
for ii, (m,s,coor) in enumerate(zip(confM_FBT_mean_,confM_FBT_std_,coors)):
    ax_signal_CM_.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
ax_signal_CM_.set(xticks=(0.5,1.5),yticks=(0.75,1.75),
        xticklabels=['non spindle','spindle'],)
ax_signal_CM_.set_yticklabels(['spindle','non spindle'],rotation=90)
ax_signal_CM_.set_title('Improved FBT model',fontweight='bold',fontsize=20)
ax_signal_CM_.set_ylabel('True label',fontsize=20,fontweight='bold')
ax_signal_CM_.set_xlabel('Predicted label',fontsize=20,fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

#fig.get_axes()[0].annotate('(a)',(0.35,45.01),fontsize=26)
#fig.get_axes()[1].annotate('(b)',(0.05,1.02),fontsize=26)
#fig.get_axes()[3].annotate('(c)',(0.05,2.0),fontsize=26)
#fig.tight_layout()
fig.savefig('%sindividual performance (3 models) (edited).png'%folder, bbox_inches='tight')
fig.savefig('%sindividual performance (3 models) (high).png'%folder, bbox_inches='tight',
            dpi=300)


##############
##################
######################
###########################
##################################
FBT = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections.items():
    FBT.append(item)
ML = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_predictions_ML.items():
    ML.append(item)
FBT_a = []
for keys, (item,fpr,tpr,confM,sensitivity,specificity) in all_detections_alter.items():    
    FBT_a.append(item)
FBT, ML, FBT_a = np.array(FBT), np.array(ML), np.array(FBT_a)
from random import shuffle
from scipy.stats import percentileofscore
def Permutation_test(data1, data2, n1=100,n2=100):
    p_values = []
    for simulation_time in range(n1):
        shuffle_difference =[]
        experiment_difference = np.mean(data1) - np.mean(data2)
        vector_concat = np.concatenate([data1,data2])
        for shuffle_time in range(n2):
            shuffle(vector_concat)
            new_data1 = vector_concat[:len(data1)]
            new_data2 = vector_concat[len(data1):]
            shuffle_difference.append(np.mean(new_data1) - np.mean(new_data2))
        p_values.append(min(percentileofscore(shuffle_difference,experiment_difference)/100.,
                            (100.-percentileofscore(shuffle_difference,experiment_difference))/100.))
    
    return p_values,np.mean(p_values),np.std(p_values)
FF, FF_mean, FF_std = Permutation_test(FBT_a.mean(1), FBT.mean(1))
FM, FM_mean, FM_std = Permutation_test(FBT_a.mean(1), ML.mean(1))
























