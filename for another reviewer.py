# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 14:03:56 2017

@author: ning
"""

#import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import combinations
import eegPinelineDesign
import mne
import seaborn as sns
import itertools
sns.set_style('white')
from Filter_based_and_thresholding import Filter_based_and_thresholding
from sklearn import metrics
working_dir = 'D:\\NING - spindle\\training set\\re-run\\'
file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
df = pd.read_csv(working_dir + 'optimized.csv')
df = df.dropna()
test_sets =np.array([[5,6,8,9],[10,11],[12,13,14],[15,16],[17,18],[19,20,21,22],[28,29,30]])
results = {'sub':[],'day':[],'validation':[],'CM':[]}
for ii,test in enumerate(test_sets):
    training = np.delete(test_sets,ii)
    print(list(itertools.chain(*training)))
    ############### training block ################################
    working_df = []
    for sub in list(itertools.chain(*training)):
        temp = df[df['sub'] == 'suj%d'%sub]
        working_df.append(temp)
    working_df = pd.concat(working_df)
    params = working_df.groupby(['higher_threshold','lower_threshold'])['roc_auc'].mean().reset_index()
    params = params.iloc[np.argmax(params['roc_auc'].values)][['higher_threshold','lower_threshold']].values
    ################### finish training ##############################
    ################## start testing ####################################
    for test_sub in test: # for each test subject in the test set
        raw_files = [f for f in list_file_to_read if ('suj%d'%test_sub in f)]
        for raw_file in raw_files:
            if test_sub > 10:
                sub_ = test_sub
                day = raw_file.split('_')[-1][3]
                annotation_file = [f for f in annotation_in_fold if ('suj%d'%sub_ in f) and ('day%s'%day in f)]
                raw = mne.io.read_raw_fif(raw_file,preload=True,)
                raw.resample(500)
            else:
                sub_ = test_sub
                day = raw_file.split('_')[1][1]
                annotation_file = [f for f in annotation_in_fold if ('suj%d'%sub_ in f) and ('d%s'%day in f)]
                raw = mne.io.read_raw_fif(raw_file,preload=True)
            try:
                annotations = pd.read_csv(annotation_file[0])
                ########## the model ###########################
                a=Filter_based_and_thresholding()
                a.get_raw(raw)
                a.get_epochs()
                a.get_annotation(annotations)
                a.mauanl_label()
                higher_threshold,lower_threshold = params
                a.find_onset_duration(lower_threshold,higher_threshold)
                a.prepare_validation()
                a.sleep_stage_check()
                # finish the model #############################
                #a.fit()
                results['sub'].append(sub_)
                results['day'].append(int(day))
                results['validation'].append(metrics.roc_auc_score(a.manual_labels,a.auto_label))
                results['CM'].append(metrics.confusion_matrix(a.manual_labels,a.auto_label))
            except:
                pass
results = pd.DataFrame(results)                
results.to_csv(working_dir + 'cross validation (review).csv',index=False)    
###############################################################################################
results = pd.read_csv(working_dir + 'cross validation (review).csv')
CMs = results['CM'].values
df = {'TN':[],'FN':[],'FP':[],'TP':[]}
import re
for CM in CMs:
    value_extract = re.findall('\d+',CM)
    CM = np.array(value_extract,dtype=int).reshape(2,2)
    CM = CM / CM.sum(axis=1)[:, np.newaxis]
    TN,FN,FP,TP = CM.flatten()
    df['TN'].append(TN)
    df['FN'].append(FN)
    df['FP'].append(FP)
    df['TP'].append(TP)
df = pd.DataFrame(df)
for name in df.columns:
    results[name] = df[name]
results = results[['sub','day','validation','TN','FN','FP','TP']]
mean_table = results.groupby(['sub','day']).mean().reset_index()
df_plot = {'sub':[],'value':[],'attribution':[]}
names = ['validation','TN','FN','FP','TP']
cmnt = ['ROC AUC','True negative rate','False negative rate','False positive rate','True positive rate']
for _,row in results.iterrows():
    sub=row['sub']
    day=row['day']
    working_df = results[(results['sub'] == sub) & (results['day'] == day)]
    for _,each_row in working_df.iterrows():
        
        for name, attri in zip(names,cmnt):
            #print(name,attri)
            df_plot['sub'].append('subject%d'%sub+'_'+'day%d'%day)
            df_plot['attribution'].append(attri)
            df_plot['value'].append(each_row[name])
df_plot = pd.DataFrame(df_plot)
df_plot.to_csv(working_dir+'to plot interpersonal variability.csv',index=False)

g=sns.factorplot(x='sub',y='value',hue='attribution',data=df_plot,size=5,aspect=4,legend_out=False)
g.set_xticklabels(rotation=45)
g.map(plt.hlines,y=0.5,xmin=0,xmax=mean_table.shape[0],colors='k',linestyle='--',)
g.map(plt.hlines,y=0.33,xmin=0,xmax=mean_table.shape[0],colors='k',linestyle='--',)
g.set_xlabels('Subject and day')
g.set_ylabels('Mean values')
g.set_titles('Interpersonal variability')
plt.legend(loc='upper right')
g.savefig(working_dir+'interpersonal variability.png',dpi=500)

for_comparison = []
for sub in pd.unique(results['sub']):
    temp_df = results[results['sub']==sub]
    if pd.unique(temp_df['day']).shape[0] > 1:
        for_comparison.append(temp_df)
for_comparison = pd.concat(for_comparison)
for_comparison.to_csv(working_dir+'comparing day 1 and day 2.csv',index=False)
day1 = results[results['day']==1]
day2 = results[results['day']==2]
comparisons = {'mean':[],'std':[],'attribution':[]}
for name in results.columns[2:]:
    a = day1[name].values
    b = day2[name].values
    _,mean,std = eegPinelineDesign.Permutation_test(a,b,)
    comparisons['mean'].append(mean)
    comparisons['std'].append(std)
    comparisons['attribution'].append(name)
comparisons = pd.DataFrame(comparisons)
comparisons.to_csv(working_dir+'comparing results.csv',index=False)





















