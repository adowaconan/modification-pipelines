# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:46:35 2017

@author: install
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import eegPinelineDesign
import pickle
from sklearn.metrics import auc
try:
    file_in_fold = eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\allData')
except:
    file_in_fold = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\allData')
spindle_type ='fast'
with open('%s_validations.p'%spindle_type, 'rb') as handle:
    validations = pickle.load(handle)
thresholds = np.arange(0.1,.85,.05);syn_channels = [1,2,3,4,5,6]
plt.close('all')
names = 'tn,fp,fn,tp,precision,recall,fbeta_score,tpr,fpr,matthews_corrcoef'.split(',')
#confusion matrix
fig, axes = plt.subplots(2,2,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
for ii,ax in enumerate(axes.flat):
    T = ax.pcolormesh(xx,yy,np.nanmean(validations[names[ii]],axis=2),cmap=plt.cm.coolwarm)
    fig.colorbar(T,ax=ax)
    ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='%s'%names[ii])
"""
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,np.nanmean(tpr,axis=2))
plt.colorbar(T)
ax.set(xticks=thresholds[::3],xticklabels=thresholds[::3],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='%s, thresholds and number of channels'%'tpr')
#confusion matrix
fig, axes = plt.subplots(2,2,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
for ii,ax in enumerate(axes.flat):
    T = ax.pcolormesh(xx,yy,np.nanmean(validations[names[ii]],axis=2),cmap=plt.cm.coolwarm)
    fig.colorbar(T,ax=ax)
    ax.set(xticks=thresholds[::5],xticklabels=thresholds[::5],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='%s'%names[ii])
fig,ax = plt.subplots(1,1,figsize=(20,20))
xx,yy = np.meshgrid(thresholds,syn_channels)
T = ax.pcolormesh(xx,yy,np.nanmean(validations['precision'],axis=2),cmap=plt.cm.coolwarm)
plt.colorbar(T)
ax.set(xticks=thresholds[::5],xticklabels=thresholds[::5],
       yticklabels=syn_channels,
       xlim=(thresholds.min(),thresholds.max()),ylim=(1,6),
       xlabel='proportions above the mean',ylabel='numbers of channels required to pass the threshold',
       title='%s'%'precision')
       
fig,ax = plt.subplots(figsize=(20,20))
fpr = validations['fpr']
tpr = validations['tpr']
fpr_mean = np.mean(fpr,axis=2)
fpr_se  = np.std( fpr,axis=2)/np.sqrt(10)
tpr_mean = np.mean(tpr,axis=2)
tpr_se  = np.std(  tpr,axis=2)/np.sqrt(10)
for ii in range(6):
    fpr = np.sort(np.concatenate(([0],fpr_mean[ii,:],[1])))
    idx = np.argsort(np.concatenate(([0],fpr_mean[ii,:],[1])))
    fprSE = np.concatenate(([0],fpr_se[ii,:],[0]))[idx]
    
    tpr = np.sort(np.concatenate(([0],tpr_mean[ii,:],[1])))
    idxx= np.argsort(np.concatenate(([0],tpr_mean[ii,:],[1])))
    tprSE = np.concatenate(([0],tpr_se[ii,:],[0]))[idxx]
    
    ax.errorbar(fpr, tpr, marker='.',fmt='',alpha=1.,label='%d channel, AUC = %.3f'%(ii+1,auc(fpr,tpr)))
    ax.fill_between(fpr,tpr+tprSE,tpr-tprSE,alpha=0.2,color='red')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',alpha=0.1)    
ax.legend(loc='best')
ax.set(xlabel='false positive rate',ylabel='true positive rate',
       title='ROC curve, 10 subjects,shaded standard error with red')
"""