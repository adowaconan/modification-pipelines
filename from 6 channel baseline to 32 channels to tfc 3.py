# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:30:42 2017

@author: ning
"""

import mne
from tqdm import tqdm
from collections import Counter

from sklearn.pipeline import make_pipeline,make_union,Pipeline
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer,StandardScaler
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_predict
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn import metrics
import pickle
from scipy import interp
import os
os.chdir('D:/Ning - spindle/')
import eegPinelineDesign
import seaborn as sns
from eegPinelineDesign import getOverlap#thresholding_filterbased_spindle_searching
from Filter_based_and_thresholding import Filter_based_and_thresholding

from matplotlib import pyplot as plt
from scipy import stats
from mne.time_frequency import tfr_multitaper,tfr_morlet
from mne.decoding import Vectorizer

import pandas as pd
import re
import numpy as np
os.chdir('D:\\NING - spindle\\training set\\') # change working directory
saving_dir='D:\\NING - spindle\\training set\\increase_channel_number\\'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)
annotation_in_fold = [f for f in os.listdir() if ('annotations.txt' in f)] # get all the possible annotation files
list_file_to_read = [f for f in os.listdir() if ('raw_ssp.fif' in f)] # get all the possible preprocessed data, might be more than or less than annotation files

cv = StratifiedShuffleSplit(n_splits=5,train_size=0.75,test_size=0.25,random_state=12345)

#channel_6,channel_32,channel_61 = {},{},{}
k = pickle.load(open(saving_dir + 'model comparison.p','rb'))
channel_6,channel_32,channel_61 = k

#for file in list_file_to_read[1:]:
#    if file == 'suj20_l2nap_day2_raw_ssp.fif':
#        pass
#    elif file == 'suj12_l2nap_day1_raw_ssp.fif':
#        pass
#    else:
#        try:
#            sub,_,day = re.findall('\d+',file)
#        except:
#            sub,day = re.findall('\d+',file)
#        print(sub,day)
#        for_name = 'sub%s_day%s' % (sub,day)
#        
#        annotation_file = [item for item in annotation_in_fold if ('suj%s_'%sub in item) and (day in item)]
#        print(file,annotation_file)
#        if len(annotation_file) != 0:
#            annotation = pd.read_csv(annotation_file[0])
#            raw = mne.io.read_raw_fif(file,preload=True)
#            
#            
#            picks = mne.pick_types(raw.info,eeg=True,meg=False,eog=False)
#            raw.filter(11,16,picks=picks)
#            raw_6 = raw.copy()
#            raw_32 = raw.copy()
#            raw_61 = raw.copy()
#            raw_6.pick_channels(['F3','F4','C3','C4','O1','O2'])
#            raw_32.pick_channels(raw.ch_names[:32])
#            raw_61.pick_channels(raw.ch_names[:61])
#            raw_6.info.normalize_proj()
#            raw_32.info.normalize_proj()
#            raw_61.info.normalize_proj()
#            
#            del raw
#            
#            r6 = eegPinelineDesign.detection_pipeline_crossvalidation(raw_6,raw_6.ch_names,
#                                                                    annotation,1000,
#                                                                    0.4,3.4,3,
#                                                                    11,16,annotation_file,cv=cv,
#                                                                    auc_threshold='adapt')
#            channel_6[for_name]=r6
#            r32 = eegPinelineDesign.detection_pipeline_crossvalidation(raw_32,raw_32.ch_names,
#                                                                    annotation,1000,
#                                                                    0.4,3.4,16,
#                                                                    11,16,annotation_file,cv=cv,
#                                                                    auc_threshold='adapt')
#            channel_32[for_name]=r32
#            r61 = eegPinelineDesign.detection_pipeline_crossvalidation(raw_61,raw_61.ch_names,
#                                                                    annotation,1000,
#                                                                    0.4,3.4,int(61/2),
#                                                                    11,16,annotation_file,cv=cv,
#                                                                    auc_threshold='adapt')
#            channel_61[for_name]=r61
#            print(for_name,np.mean(r6[3],0),np.mean(r32[3],0),np.mean(r61[3],0))
#            k = [channel_6,channel_32,channel_61]
#            pickle.dump(k,open(saving_dir + 'model comparison.p','wb'))


"""auc,fpr,tpr, confusion matrix, sensitivity, specificity"""
cf6,cf32,cf61 = [], [], []
auc6,auc32,auc61 = [],[],[]
for sub in channel_6.keys():
    r6,r32,r61 = channel_6[sub],channel_32[sub],channel_61[sub]
    cf6_,cf32_,cf61_ = np.mean(r6[-3],0),np.mean(r32[-3],0),np.mean(r61[-3],0)
    cf6.append(cf6_);cf32.append(cf32_);cf61.append(cf61_)
    
    auc6_,auc32_,auc61_ = r6[0],r32[0],r61[0]
    auc6.append(auc6_);auc32.append(auc32_);auc61.append(auc61_)

auc6,auc32,auc61 = np.array(auc6),np.array(auc32),np.array(auc61) 

cf6_mean,cf32_mean,cf61_mean = np.mean(cf6,0),np.mean(cf32,0),np.mean(cf61,0)
cf6_se,cf32_se,cf61_se = np.std(cf6,0)/np.sqrt(len(channel_6.keys())),np.std(cf32,0)/np.sqrt(len(channel_6.keys())),np.std(cf61,0)/np.sqrt(len(channel_6.keys()))

auc6_mean,auc32_mean,auc61_mean=np.mean(auc6,0),np.mean(auc32,0),np.mean(auc61,0)
auc6_se,auc32_se,auc61_se=np.std(auc6,0)/np.sqrt(len(channel_6.keys())),np.std(auc32,0)/np.sqrt(len(channel_6.keys())),np.std(auc61,0)/np.sqrt(len(channel_6.keys()))


"""left subplot, auc roc scores"""
yy_axis_labels = list(channel_6.keys())
yy_axis_labels = [keys.split('_')[0][3:] +'          ' + keys.split('_')[1][-1] + ' ' for keys in yy_axis_labels]
yy_axis = np.arange(len(yy_axis_labels)+1)
idx_sort = np.argsort(auc6.mean(1))
sort_ylabel = list(np.array(yy_axis_labels)[idx_sort])
sort_ylabel.append('Subject      Day')

fig= plt.figure(figsize=(24,16));cnt = 0;uv=9.7
ax = fig.add_subplot(131)
for ii,(auc_,color,lge) in enumerate(zip([auc6,auc32,auc61],['blue','red','green'],['6 channels','32 channels','61 channels'])):
    ax.errorbar(x=auc_.mean(1)[idx_sort],y=yy_axis[:-1]+(ii/10),xerr=auc_.std(1)[idx_sort]/np.sqrt(5),linestyle='',color=color,label=lge)
    ax.axvspan(ymax=len(yy_axis_labels)/(len(yy_axis_labels)+uv),
               xmin=auc_.mean()-auc6.std()/np.sqrt(len(channel_6.keys())),
               xmax=auc_.mean()+auc6.std()/np.sqrt(len(channel_6.keys())),
               color=color,alpha=0.4,
               label='mean AUC, %s,%.2f$\pm$%.2f'%(lge,auc_.mean(),auc_.std()/np.sqrt(len(channel_6.keys()))))
ax.set(yticks=yy_axis,yticklabels=sort_ylabel,title='AUC ROC scores',
       ylim=(-0.5,len(yy_axis_labels)+uv),xlabel='AUC ROC scores')   
ax.legend(loc='upper left')

for ii,(dict_,color,lge,position) in enumerate(zip([channel_6,channel_32,channel_61],['blue','red','green'],['6 channels','32 channels','61 channels'],[2,5,8])):
    ax=fig.add_subplot(3,3,position)
    temp_fpr,temp_tpr = [],[]
    sub_pick = np.array(list(dict_.keys()))[idx_sort][int(41/2)]
    for key,items in dict_.items():
        if key != sub_pick:
            temp_fpr.append(items[1])
            temp_tpr.append(items[2])
    temp_fpr,temp_tpr = np.array(temp_fpr),np.array(temp_tpr)
    for fpr,tpr in zip(temp_fpr.flatten(),temp_tpr.flatten()):
        ax.plot(fpr,tpr,alpha=0.05,color=color,)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    items = dict_[sub_pick]
    fpr,tpr = items[1],items[2]
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
    keys = sub_pick.split('_')
    ax.plot(base_fpr, mean_tprs,
               label='Most median sample\nsubject %s, day %s\nAUC: %.2f $\pm$ %.4f'%(keys[0],keys[1],
                                                                                     np.mean(items[0]),
                                                                                     np.std(items[0])/np.sqrt(5)),
               color='black',alpha=1.)
    ax.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    ax.plot([0,1],[0,1],linestyle='--',color='black',alpha=0.8)
    ax.legend(loc='lower right')
    ax.set(title='%s'%lge,xlim=(0,1),ylim=(0,1))
cf6,cf32,cf61 = np.array(cf6),np.array(cf32),np.array(cf61)
for ii,(data,color,lge,position) in enumerate(zip([cf6,cf32,cf61],['blue','red','green'],['6 channels','32 channels','61 channels'],[3,6,9])):
    ax=fig.add_subplot(3,3,position)
    
    ax=sns.heatmap(data.mean(0).reshape(2,2),cbar=False,cmap=plt.cm.Blues,
                         vmin=0,vmax=1.,
                         ax=ax,annot=False,
                         xticklabels=['non spindle','spindle'],yticklabels=['non spindle','spindle']) 
    coors = np.array([[0,0],[1,0],[0,1],[1,1],])+ 0.5
    for ii, (m,s,coor) in enumerate(zip(data.mean(0),data.std(0)/np.sqrt(5),coors)):
        ax.annotate('%.2f $\pm$ %.2f'%(m,s),xy = coor,size=25,weight='bold',ha='center')
    #ax.set(xticks=(0.5,1.5),yticks=(0.5,1.5),
    #        xticklabels=['non spindle','spindle'],yticklabels=['non spindle','spindle'])
    #ax.set_yticklabels(['non spindle','spindle'],)#rotation=90)
    ax.set_title('Between subject confusion matrix\n%s'%(lge),fontweight='bold',fontsize=20)
    ax.set_ylabel('True label',fontsize=20,fontweight='bold')
fig.tight_layout()

fig.savefig(saving_dir+'increase channels.png',dpi=500)































#ax.errorbar(x=auc32.mean(1)[idx_sort],y=yy_axis[:-1]+0.1,xerr=auc32.std(1)[idx_sort]/np.sqrt(5),linestyle='',color='red',label='32 channels')
#ax.errorbar(x=auc61.mean(1)[idx_sort],y=yy_axis[:-1]+0.2,xerr=auc61.std(1)[idx_sort]/np.sqrt(5),linestyle='',color='green',label='61 channels')
#ax.axvspan(ymax=len(yy_axis_labels)/(len(yy_axis_labels)+uv),
#           xmin=auc32.mean()-auc32.std()/np.sqrt(len(channel_6.keys())),
#           xmax=auc32.mean()+auc32.std()/np.sqrt(len(channel_6.keys())),
#           color='red',alpha=0.4,
#           label='mean AUC, 32 channels,%.2f$\pm$%.2f'%(auc32.mean(),auc32.std()/np.sqrt(len(channel_6.keys()))))
#ax.axvspan(ymax=len(yy_axis_labels)/(len(yy_axis_labels)+uv),
#           xmin=auc61.mean()-auc61.std()/np.sqrt(len(channel_6.keys())),
#           xmax=auc61.mean()+auc61.std()/np.sqrt(len(channel_6.keys())),
#           color='green',alpha=0.4,
#           label='mean AUC, 61 channels,%.2f$\pm$%.2f'%(auc61.mean(),auc61.std()/np.sqrt(len(channel_6.keys()))))