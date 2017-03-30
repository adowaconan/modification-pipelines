# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:13:47 2017

@author: install
"""

import mne
import eegPinelineDesign
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os


file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
l,h = (11,16);
low, high=11,16
folder='step_size_500_%d_%dgetting_higher_threshold\\'%(l,h)
if not os.path.exists(folder):
    os.makedirs(folder)

# over thresholds 
windowSize=500;6;syn_channel=3
with_sleep_stage,without_sleep_stage,wavelet={},{},{}
for threshold in np.arange(0.1,1.1,0.1):
    for higher_threshold in np.arange(2,3.6,0.1):
        temp_with_sleep_stage,temp_without_sleep_stage,temp_wavelet={},{},{};
        with_stage_samples,with_stage_label=[],[]
        without_stage_samples,without_stage_label=[],[]
        with_stage_wavelet_samples,with_stage_wavelet_label=[],[]
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
                print('gathering...................')
                annotation = pd.read_csv(annotation_file[0])
                ################### eeg data part ###########################
                raw = mne.io.read_raw_fif(file,preload=True)
                if old:
                    pass
                else:
                    raw.resample(500, npad="auto") # down sampling Karen's data
                raw.pick_channels(channelList)
                raw.filter(low,high);print('finish preprocessing................')
                ######## get one done #####with sleep stage info#####
                print('with sleep stage info............')
                print('take out first 300 seconds and last 100 seconds............')
                temp_with_sleep_stage,with_stage_samples,with_stage_label=eegPinelineDesign.data_gathering_pipeline(temp_with_sleep_stage,
                                with_stage_samples,
                                with_stage_label,do='with_stage',sub=sub,day=day,raw=raw,channelList=channelList,
                                file=file,windowSize=windowSize,
                                threshold=threshold,syn_channel=syn_channel,
                                l=l,h=h,annotation=annotation,old=old,annotation_file=annotation_file,
                                higher_threshold=higher_threshold)
                """
                ####### get one done #### without sleep stage #####
                print('without sleep stage info...........')
                temp_without_sleep_stage,without_stage_samples,without_stage_label=eegPinelineDesign.data_gathering_pipeline(temp_without_sleep_stage,
                                without_stage_samples,
                                without_stage_label,do='without_stage',sub=sub,day=day,raw=raw,channelList=channelList,
                                file=file,windowSize=windowSize,
                                threshold=threshold,syn_channel=syn_channel,
                                l=l,h=h,annotation=annotation,old=old,annotation_file=annotation_file,
                                higher_threshold=higher_threshold)
                
                print('with wavelet transform...........')
                ####### get one done ## with sleep stage and wavelet transform######
                temp_wavelet,with_stage_wavelet_samples,with_stage_wavelet_label=eegPinelineDesign.data_gathering_pipeline(temp_wavelet,
                                with_stage_wavelet_samples,
                                with_stage_wavelet_label,'wavelet',sub,day,raw=raw,channelList=channelList,
                                file=file,windowSize=windowSize,
                                threshold=threshold,syn_channel=syn_channel,
                                l=l,h=h,annotation=annotation,old=old,annotation_file=annotation_file,
                                higher_threshold=higher_threshold)
                
                """
                raw.close()
                plt.close('all')
            else:
                print(sub+day+'no annotation')
        with_sleep_stage[str(threshold)+','+str(higher_threshold)]=temp_with_sleep_stage;
        #without_sleep_stage[str(threshold)+','+str(higher_threshold)]=temp_without_sleep_stage
        #wavelet[threshold]=temp_wavelet
        print('saving temporal datas')
        pickle.dump(with_stage_samples,open("%s%.2f-%.2f_samples_with.p"%(folder,threshold,higher_threshold),"wb"))
        pickle.dump(with_stage_label,  open("%s%.2f-%.2f_labels_with.p" %(folder,threshold,higher_threshold),"wb"))
        #pickle.dump(without_stage_samples,open("%s%.2f-%.2f_samples_without.p"%(folder,threshold,higher_threshold),"wb"))
        #pickle.dump(without_stage_label,  open("%s%.2f-%.2f_labels_without.p" %(folder,threshold,higher_threshold),"wb"))
        """
        pickle.dump(with_stage_wavelet_samples,open("temp_data\\%.2f_samples_with_wavelet.p"%threshold,"wb"))
        pickle.dump(with_stage_wavelet_label,open("temp_data\\%.2f_labels_with_wavelet.p"%threshold,"wb"))
        """

        over_threshold={'with':with_sleep_stage}#,'without':without_sleep_stage}
        pickle.dump( over_threshold, open( "%sover_threshold.p"%folder, "wb" ) )

"""
from random import shuffle
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
with open('%sover_threshold.p'%folder,'rb') as h:
    over_threshold = pickle.load(h)
types = 'with'
df_with = eegPinelineDesign.compute_measures(over_threshold,types,plot_flag=True)
threshold = df_with['thresholds'][np.argmax(df_with['AUC'].mean(1))]

samples = pickle.load(open("%s%.2f_samples_%s.p"%(folder,threshold,types),
                                "rb"))
label = pickle.load(open("%s%.2f_labels_%s.p" %(folder,threshold,types),
                        "rb"))
data,label,idx_row = np.array(samples),np.array(label),np.arange(0,len(label),1)
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.95)
tpot = TPOTClassifier(generations=5, population_size=20,
                      verbosity=2,random_state=0,num_cv_folds=3 )
tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)
tpot.export('%s%s_tpot_exported_pipelien.py'%(folder,types) )  


types = 'without'
df_without = eegPinelineDesign.compute_measures(over_threshold,types,plot_flag=True)
threshold = df_without['thresholds'][np.argmax(df_without['AUC'].mean(1))]

samples = pickle.load(open("temp_data\\%.2f_samples_%s.p"%(threshold,types),
                                "rb"))
label = pickle.load(open("temp_data\\%.2f_labels_%s.p" %(threshold,types),
                        "rb"))
data,label,idx_row = np.array(samples),np.array(label),np.arange(0,len(label),1)
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.85)
tpot = TPOTClassifier(generations=5, population_size=20,
                      verbosity=2,random_state=0,num_cv_folds=3 )
tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)
tpot.export('%s_tpot_exported_pipelien.py'%types)  
"""