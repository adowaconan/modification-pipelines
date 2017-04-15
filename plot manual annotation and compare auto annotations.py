# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:45:59 2017

@author: Ning
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import eegPinelineDesign
import mne
import re
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
plt.rc('font', size=26)  
matplotlib.rc('axes', titlesize=30)
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
import warnings
warnings.filterwarnings("ignore")

file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=500;threshold=0.4;syn_channel=3
l,h = (11,16);
low, high=11,16
hh=3.5
front=300;back=100;total=front+back
if False:
    for file in list_file_to_read:
        sub = file.split('_')[0]
        if int(sub[3:]) >= 11:
            day = file.split('_')[2][:4]
            day_for_show = day
            old = False
        else:
            day = file.split('_')[1]
            day_for_show = day[0]+'ay'+day[1]
            old = True
    
        annotation_file = [item for item in annotation_in_fold if (sub in item) and (day in item)]
        if len(annotation_file) != 0:
            annotations = pd.read_csv(annotation_file[0])
            raw = mne.io.read_raw_fif(file,preload=True)
            if old:
                pass
            else:
                raw.resample(500, npad="auto") # down sampling Karen's data
            raw.pick_channels(channelList)
            raw.filter(low,high)
            time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw=raw,channelList=channelList,
                                                    annotations=annotations,
                                                    moving_window_size=windowSize,threshold=threshold,
                                            syn_channels=syn_channel,l_freq=l,h_freq=h,higher_threshold=hh)
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
            result = result[result.Onset > front]
            result.to_csv(sub+"_"+day_for_show+"auto_annotation.csv")

            
auto_annotation = [f for f in file_in_fold if ('auto_annotation.csv' in f)]
cnt_old = 0;cnt_new=0;all_sub={'new':[],'old':[]}
manual_only={'new':[],'old':[]}
for file in list_file_to_read:
    sub = file.split('_')[0]
    if int(sub[3:]) >= 11:
        day = file.split('_')[2][:4]
        day_for_show = day
        old = False
    else:
        day = file.split('_')[1]
        day_for_show = day[0]+'ay'+day[1]
        old = True

    manual = [item for item in annotation_in_fold if (sub in item) and (day in item)]
    if len(manual) != 0:
        raw = mne.io.read_raw_fif(file,preload=True)
        auto = pd.read_csv(sub+"_"+day_for_show+"auto_annotation.csv")
        manual_spindle = pd.read_csv(manual[0])
        manual_spindle = manual_spindle[manual_spindle.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
        manual_spindle = manual_spindle[manual_spindle.Onset > front]
        keyword = re.compile('spindle',re.IGNORECASE)
        gold_standard = {'Onset':[],'Annotation':[]}
        for ii,row in manual_spindle.iterrows():
            if keyword.search(row[-1]):
                gold_standard['Onset'].append(float(row.Onset))
                gold_standard['Annotation'].append(row.Annotation)
        gold_standard = pd.DataFrame(gold_standard) 
        auto['Spindle']=['Automated']*len(auto)
        gold_standard['Spindle']=['Manual']*len(gold_standard)
        auto=auto[['Onset','Annotation','Spindle']]
        auto['Subject']=['Subject '+sub[3:]+"_"+day_for_show]*len(auto)
        gold_standard['Subject']=['Subject '+sub[3:]+"_"+day_for_show]*len(gold_standard)
        if old:
            all_sub['old'].append(auto)
            all_sub['old'].append(gold_standard)
        else:
            all_sub['new'].append(auto)
            all_sub['new'].append(gold_standard)
        
            
        gold_standard['Sub']=['Subject '+sub[3:]]*len(gold_standard)
        gold_standard['day']=[day_for_show]*len(gold_standard)
        if old:
            manual_only['old'].append(gold_standard)
        else:
            manual_only['new'].append(gold_standard)

sns.set_style("white")
order=['Subject 5_day2',
       'Subject 6_day1', 'Subject 6_day2', 'Subject 8_day1',
       'Subject 8_day2', 'Subject 9_day1', 'Subject 9_day2',
       'Subject 10_day1', 'Subject 10_day2']            
fig, ax = plt.subplots(figsize=(20,25),nrows=2)        
new = pd.concat(all_sub['new'])
old = pd.concat(all_sub['old'])
ax[0]=sns.violinplot(y='Subject',x='Onset',hue='Spindle',data=old,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[0],scale='area',scale_hue=True,
                    order=order,palette={"Automated": "#2976bb", "Manual": "#20c073"})
ax[0].set(xlim=(0,4000),xlabel='',
            ylabel='')  
ax[0].set_title('Long recordings',fontweight='bold')
lgd1=ax[0].legend(prop={'size':26})
ax[1]=sns.violinplot(y='Subject',x='Onset',hue='Spindle',data=new,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[1],scale='area',scale_hue=True,
                palette={"Automated": "#2976bb", "Manual": "#20c073"})
ax[1].set(xlim=(0,2000),xlabel='Time (Sec)',
            ylabel='')
ax[1].set_title('Short recordings',fontweight='bold')
lgd2=ax[1].legend(loc='best',prop={'size':20})
fig.tight_layout()       
fig.savefig('manu vs auto.png')       
        
        
#manual_only = pd.concat(manual_only)
order=['Subject 5', 'Subject 6', 'Subject 8', 'Subject 9','Subject 10']
f, ax = plt.subplots(figsize=(20,25),nrows=2)        
new = pd.concat(manual_only['new'])
old = pd.concat(manual_only['old'])
ax[0]=sns.violinplot(y='Sub',x='Onset',hue='day',data=old,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[0],scale='area',scale_hue=True,
                    order=order,palette={"day1": "#2976bb", "day2": "#20c073"})
ax[0].set(xlim=(0,4000),xlabel='',
            ylabel='')
ax[0].set_title('Long recordings',fontweight='bold')  
lgd1=ax[0].legend(prop={'size':28})  
ax[1]=sns.violinplot(y='Sub',x='Onset',hue='day',data=new,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[1],scale='area',scale_hue=True,
                    palette={"day1": "#2976bb", "day2": "#20c073"})  
ax[1].set(xlim=(0,2000),xlabel='Time (Sec)',
            ylabel='')
handles, labels = ax[1].get_legend_handles_labels()
ax[1].set_title('Short recordings',fontweight='bold')
lgd2=ax[1].legend(handles[::-1], labels[::-1],loc='best',prop={'size':28})
fig.tight_layout()
f.savefig('manu only.png',bbox_inches = 'tight')      
        
subjects=[11,12,13,14,15,16,17,18,19,20,21,22,23,26,27,28,29,30,32] # missing 12, 20, 25
slow = 10,12   
slow_count=[]
fast = 12.5,14.5
fast_count=[]    

for ii in subjects:
    try:
        raw_file = [file for file in list_file_to_read if (str(ii) in file) and ('l2' in file) and ('fif' in file)]
        print(raw_file[0])
        raw = mne.io.read_raw_fif(raw_file[0],preload=True)
        raw.pick_channels(channelList)
        day = raw_file[0].split('_')[-1][:4]
        raw.filter(slow[0],slow[1])
        anno_name =[file for file in annotation_in_fold if (str(ii) in file) and (day in file) ]
        annotations=pd.read_csv(anno_name[0])
        time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw=raw,channelList=channelList,
                                                annotations=annotations,
                                                moving_window_size=windowSize,threshold=threshold,
                                        syn_channels=syn_channel,l_freq=slow[0],h_freq=slow[1],higher_threshold=hh)
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
        result = result[result.Onset > front]    
        slow_count.append([ii,len(result['Onset']),len(result['Onset'])/(raw.last_samp/raw.info['sfreq'] - total)]) 
    except:
        pass
for ii in subjects:
    try:
        raw_file = [file for file in list_file_to_read if (str(ii) in file) and ('l2' in file) and ('fif' in file)]
        print(raw_file[0])
        raw = mne.io.read_raw_fif(raw_file[0],preload=True)
        raw.pick_channels(channelList)
        day = raw_file[0].split('_')[-1][:4]
        raw.filter(fast[0],fast[1])
        anno_name =[file for file in annotation_in_fold if (str(ii) in file) and (day in file) ]
        annotations=pd.read_csv(anno_name[0])
        time_find,mean_peak_power,Duration,peak_time,peak_at=eegPinelineDesign.spindle_validation_with_sleep_stage(raw=raw,channelList=channelList,
                                                annotations=annotations,
                                                moving_window_size=windowSize,threshold=threshold,
                                        syn_channels=syn_channel,l_freq=fast[0],h_freq=fast[1],higher_threshold=hh)
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
        result = result[result.Onset > front]    
        fast_count.append([ii,len(result['Onset']),len(result['Onset'])/(raw.last_samp/raw.info['sfreq'] - total)])
    except:
        pass        
        
        
        
fast_count = pd.DataFrame(fast_count,columns=['subject','fast spindle count','fast spindle density'])        
slow_count = pd.DataFrame(slow_count,columns=['subject','slow spindle count','slow spindle density'])
fast_count.to_csv('fast_spindle_info.csv')
slow_count.to_csv('slow_spindle_info.csv')
data = pd.read_clipboard()
vars=['WM', 'REC1', 'REC2', 'Sleep Latency', 'Total Nap time',
       'Spindle Density (Karen)', 'Spindle Count (Karen)',
       'Slow Spindle Count (Ning)', 'Slow Spindle Density (Ning)',
       'Fast Spindle Count (Ning)', 'Fast Spindle Density (Ning)',
       'Cluster 1 % TSE', 'Cluster 2 % TSE', 'Cluster 3 % TSE']
sns.pairplot(data,vars=['WM','Slow Spindle Count (Ning)'],diag_kind='kde')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParamsDefault.update({'axes.labelsize':320,'xtick.labelsize':32,'ytick.labelsize':32})
def standardized(x):
    return (x - x.mean()) / x.std(ddof=0)
var_list = ['WM', 'REC1', 'REC2', 
       'Spindle Count (Karen)',
       'Slow Spindle Count (Ning)',
       'Fast Spindle Count (Ning)',
       'Cluster 1 % TSE', 'Cluster 2 % TSE', 'Cluster 3 % TSE']
for v in var_list:
    data[v] = standardized(data[v])
var_list2= ['Slow Spindle Count (Ning)',
       'Fast Spindle Count (Ning)',
       'Cluster 1 % TSE', 'Cluster 2 % TSE', 'Cluster 3 % TSE']
xvar = ['Cluster 1 % TSE', 'Cluster 2 % TSE', 'Cluster 3 % TSE']
yvar = ['Slow Spindle Count (Ning)','Fast Spindle Count (Ning)']

sns.pairplot(data,x_vars=xvar,y_vars=yvar,diag_kind='kde',kind='reg',size=5,aspect=2,
             plot_kws=dict(robust =True ),)
fig, ax = plt.subplots(figsize=(10,10))
ax=sns.heatmap(data[data.columns[2:]].corr(),ax=ax)
plt.setp(ax.yaxis.get_majorticklabels(),rotation=50)

sns.regplot(xvar[0],yvar[1],data=data,robust=True)

