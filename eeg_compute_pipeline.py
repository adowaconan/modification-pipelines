# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:58:54 2017
@author: ning
"""
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats,signal
import re
from collections import defaultdict  
import json

def phase_locking_value(theta1, theta2):
    complex_phase_diff = np.exp(np.complex(0,1)*(theta1 - theta2))
    #plv = np.abs(np.sum(complex_phase_diff))/len(theta1)
    return complex_phase_diff
def intervalCheck(a,b,tol=0):#a is an array and b is a point
    return a[0]-tol <= b <= a[1]+tol
def spindle_comparison(time_interval,spindle,spindle_duration,spindle_duration_fix=True):
    if spindle_duration_fix:
        spindle_start = spindle - 0.5
        spindle_end   = spindle + 1.5
        a =  np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
    else:
        spindle_start = spindle - spindle_duration/2.
        spindle_end   = spindle + spindle_duration/2.
        a = np.logical_or((intervalCheck(time_interval,spindle_start)),
                           (intervalCheck(time_interval,spindle_end)))
        return a
def adjecency_list(con):
    graph = defaultdict(list)   
    edges=set()
    for i, v in enumerate(con, 1):
        for j, u in enumerate(v, 1):
            if u != 0 and frozenset([i, j]) not in edges:
                edges.add(frozenset([i, j]))
                graph[i].append({j: u})
    return graph
def read_annotation(raw,manual_spindle):
    
    manual_spindle = manual_spindle[manual_spindle.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
    manual_spindle = manual_spindle[manual_spindle.Onset > 100] 
    keyword = re.compile('spindle',re.IGNORECASE)
    gold_standard = {'Onset':[],'Annotation':[]}
    for ii,row in manual_spindle.iterrows():
        if keyword.search(row[-1]):
            gold_standard['Onset'].append(float(row.Onset))
            gold_standard['Annotation'].append(row.Annotation)
    gold_standard = pd.DataFrame(gold_standard) 
    return gold_standard
def discritized_onset_label_manual(raw,df,spindle_segment):
    discritized_continuous_time = np.arange(0,raw.last_samp/raw.info['sfreq'],step=spindle_segment)
    discritized_time_intervals = np.vstack((discritized_continuous_time[:-1],discritized_continuous_time[1:]))
    discritized_time_intervals = np.array(discritized_time_intervals).T
    discritized_time_to_zero_one_labels = np.zeros(len(discritized_time_intervals))
    for jj,(time_interval_1,time_interval_2) in enumerate(discritized_time_intervals):
        time_interval = [time_interval_1,time_interval_2]
        for spindle in df['Onset']:
            #print(time_interval,spindle,spindle_segment)
            if spindle_comparison(time_interval,spindle,spindle_segment):
                discritized_time_to_zero_one_labels[jj] = 1
    return discritized_time_to_zero_one_labels
def compute_plv_pli_cc(raw,duration,plv_threshold_set,pli_threshold_set,cc_threshold_set,labels,channelList,fmin=11,fmax=16):
    # make events
    event_array = mne.make_fixed_length_events(raw,id=1,duration=float(duration))
    event_array[:,-1] = np.arange(1,len(event_array)+1)
    event_array[:,1] = duration * raw.info['sfreq']
    # make epochs
    tmin, tmax = -duration*0.2, duration-duration*0.2  #20% overlapping
    epochs = mne.Epochs(raw,event_array,tmin=tmin,tmax=tmax,
                       baseline=None,preload=True,proj=False)
    sfreq = raw.info['sfreq']
    
    
    features = ['mean','variance','delta_mean',
          'delta variance','change variance',
         'activity','mobility','complexity',
         'spectral_entropy']#,'time_stamp']
    # initialize folder     
    if not os.path.exists('epoch_COH_%.2f'%duration):
        os.makedirs('epoch_COH_%.2f'%duration)
        
    if not os.path.exists('epoch_PLI_%.2f'%duration):
        os.makedirs('epoch_PLI_%.2f'%duration)
    
    if not os.path.exists('epoch_PLV_%.2f'%duration):
        os.makedirs('epoch_PLV_%.2f'%duration)
    
    
    epochFeatures = {name:[] for name in features}
    time_list=[]
    con_methods=['coh','plv','pli']
    # the entire pipeline
    for ii,epoch_data in enumerate(epochs):
        # easiest way to compute coh
        temp_connection,freqs,times,n_epochs,n_tapers=mne.connectivity.spectral_connectivity(
            epochs[str(ii+2)],
            method='coh',
            mode='multitaper',
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
                                            )
        
        temp_connection = [temp_connection[:,:,0]]
        time_list.append(epochs[str(ii+2)].events[0][0])
        print('computing features for epoch %d'%(ii+1))
        #epochFeatures['time_stamp'].append
        epochFeatures['mean'].append(np.mean(epoch_data))
        epochFeatures['variance'].append(np.var(epoch_data))
        epochFeatures['delta_mean'].append(np.mean(-np.diff(epoch_data,axis=1)))
        epochFeatures['delta variance'].append(np.var(np.mean(-np.diff(epoch_data,axis=1))))
        if ii == 0:
            epochFeatures['change variance'].append(0)
        elif ii == 1:
            epochFeatures['change variance'].append(np.mean(np.var(epoch_data - epochFeatures['mean'][ii-1])))
        else:
            epochFeatures['change variance'].append(np.mean(np.var(epoch_data - epochFeatures['mean'][ii-1] - epochFeatures['mean'][ii-1])))

        activity = np.var(epoch_data)
        epochFeatures['activity'].append(activity)
        tempData = -np.diff(epoch_data,axis=1)
        mobility = np.std(tempData)/np.sqrt(activity)
        epochFeatures['mobility'].append(mobility)

        startRange = epoch_data[:,:-2]
        endRange = epoch_data[:,2:]
        tempData = endRange - startRange
        complexity = (np.std(tempData)/(np.std(-np.diff(epoch_data,axis=1)))) /((np.std(-np.diff(epoch_data,axis=1)))/np.sqrt(activity))
        epochFeatures['complexity'].append(complexity)

        specEnt = np.zeros(shape=(6,))
        for iter in range(6):
                ampSpec=np.abs(np.fft.fft(epoch_data[0,:])) / np.sum(np.abs(np.fft.fft(epoch_data[0,:])))
                specEnt[iter]=-np.sum(ampSpec * np.log2(ampSpec))
        epochFeatures['spectral_entropy'].append(np.mean(specEnt))

        dist_list_plv = np.zeros(shape=(6,6))
        dist_list_pli = np.zeros(shape=(6,6))
        for node_1 in range(6):
            for node_2 in range(6):
                if node_1 != node_2:
                    data_1 = epoch_data[node_1,:]
                    data_2 = epoch_data[node_2,:]
                    PLV=phase_locking_value(np.angle(signal.hilbert(data_1,axis=0)),
                                             np.angle(signal.hilbert(data_2,axis=0)))
                    dist_list_plv[node_1,node_2]=np.abs(np.mean(PLV))
                    PLI=np.angle(signal.hilbert(data_1,axis=0))-np.angle(signal.hilbert(data_2,axis=0))
                    dist_list_pli[node_1,node_2]=np.abs(np.mean(np.sign(PLI)))

        temp_connection.append(dist_list_plv)        
        temp_connection.append(dist_list_pli)
        
        try:
            if labels[ii+2] ==1:
                title_label='spindle'
            else:
                title_label=''
        except:
            title_label=''
           
        #fig,ax = plt.subplots(figsize=(10,25))
        con_res=dict()
        for method, c in zip(con_methods,temp_connection):
            con_res[method] = c
        colors=plt.cm.rainbow(np.linspace(0,1,len(channelList)))
        time_plot = np.linspace(epochs[str(ii+2)].events[0][0]/raw.info['sfreq'],
                                       epochs[str(ii+2)].events[0][0]/raw.info['sfreq']+duration,
                                              epoch_data.shape[1])
        for plv_threshold,pli_threshold,cc_threshold in zip(plv_threshold_set,pli_threshold_set,cc_threshold_set):
            thresholds = {'plv':plv_threshold,'pli':pli_threshold,'coh':cc_threshold}
            for jj, method in enumerate(con_methods):
                fig,ax = plt.subplots(figsize=(16,16),nrows=2)
                mne.viz.plot_connectivity_circle(con_res[method]>thresholds[method],raw.ch_names,fig=fig,show=False,
                                                 title='%s,threshold:%.2f,%s'%(method,thresholds[method],title_label),facecolor='black',textcolor='white',
                                                                     colorbar=False,fontsize_title=22,fontsize_names=22,
                                                                     subplot=221,node_colors=colors,
                                                 )
                adjecency_df = pd.DataFrame(np.array(con_res[method]>thresholds[method],dtype=int),columns=np.arange(1,7))
                adjecency_df.to_csv('epoch_%s_%.2f\\epoch_%d_%.2f(%s).csv'%(method.upper(),duration,ii+1,thresholds[method],title_label))
                
                for kk,(name,color) in enumerate(zip(channelList,colors)):
                    ax[1].plot(time_plot,epoch_data[kk,:],label=name,color=color)
                ax[1].legend(loc='upper right')
                plt.setp(plt.getp(ax[1], 'yticklabels'), color='w') #set yticklabels color
                plt.setp(plt.getp(ax[1], 'xticklabels'), color='w') #set xticklabels color
                ax[1].set_title('%.2f-%.2f sec %s'%(time_plot.min(),time_plot.max(),title_label),color='w')
                ax[1].set_xlabel('Time',color='w')
                ax[1].set_ylabel('$\mu$V',color='w')

                fig.set_facecolor('black')
                fig.savefig('epoch_%s_%.2f\\epoch_%d_%.2f(%s).png'%(method.upper(),duration,ii+1,thresholds[method],title_label),
                                                          facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all') 
        #connection.append(temp_connection)

    epochFeatures = pd.DataFrame(epochFeatures)
    a,b=mne.time_frequency.psd_multitaper(epochs,fmin=8,fmax=16)
    epochFeatures['skewness_of_amplitude_spectrum']=np.mean(stats.skew(a,axis=2),1)
    epochFeatures['spindle']=labels[1:]
    

    return epochFeatures