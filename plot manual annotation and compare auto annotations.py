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
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
import warnings
warnings.filterwarnings("ignore")

file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
channelList = ['F3','F4','C3','C4','O1','O2']
list_file_to_read = [files for files in file_in_fold if ('fif' in files) and ('nap' in files)]
annotation_in_fold=[files for files in file_in_fold if ('txt' in files) and ('annotations' in files)]
windowSize=500;threshold=0.4;syn_channel=3
l,h = (11,16);
low, high=11,16
hh=3.4
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
            time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
                                                                                                    channelList,annotations,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=threshold,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=hh,
                                                                                                    front=300,back=100,sleep_stage=True,)
            result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
            result['Annotation'] = 'auto spindle'
            result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
            result = result[result.Onset > front]
            result.to_csv(sub+"_"+day_for_show+"auto_annotation.csv")

            
auto_annotation = [f for f in file_in_fold if ('auto_annotation.csv' in f)]
cnt_old = 0;cnt_new=0;all_sub={'new':[],'old':[]}
manual_only={'new':[],'old':[]}
for file in list_file_to_read:
    if file == 'suj20_l2nap_day2.fif':
        pass
    else:
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

import pickle
#pickle.dump(manual_only,open('manual only annotations.p','wb'))
#pickle.dump(all_sub,open('all sub.p','wb'))
manual_only = pickle.load(open('manual only annotations.p','rb'))
all_sub = pickle.load(open('all sub.p','rb'))

sns.set_style("white")
order=['Subject 5_day2',
       'Subject 6_day1', 'Subject 6_day2', 'Subject 8_day1',
       'Subject 8_day2', 'Subject 9_day1', 'Subject 9_day2',
       'Subject 10_day1', 'Subject 10_day2']            
#fig, ax = plt.subplots(figsize=(20,30),nrows=2,gridspec_kw = {'height_ratios':[9, 33]})  
import re
def reset_yticklabels(x):
    try:
        sub, day = re.findall('\d+',x)
        return sub + '        ' + day
    except:
        sub = re.findall('\d+',x)
        return int(sub[0])
fig = plt.figure(figsize=(20,25))
ax = [fig.add_subplot(221), fig.add_subplot(223)]      
new = pd.concat(all_sub['new'])
old = pd.concat(all_sub['old'])
new['Subject_'] = new['Subject'].apply(reset_yticklabels)
old['Subject_'] = old['Subject'].apply(reset_yticklabels)
yticklabels = []
for y in order:
    temp = reset_yticklabels(y)
    yticklabels.append(temp)
ax[0]=sns.violinplot(y='Subject_',x='Onset',hue='Spindle',data=old,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[0],scale='area',scale_hue=True,
                    order=yticklabels,palette={"Automated": "#2976bb", "Manual": "#20c073"})
ax[0].set(xlim=(0,4000),xlabel='',
            ylabel='')#, yticklabels=yticklabels)  
ax[0].set_title('Long recordings',fontweight='bold')
lgd1=ax[0].legend(loc='best',prop={'size':18})
# need to select data first, and before that, reset index
new = new.reset_index()
ax[1]=sns.violinplot(y='Subject_',x='Onset',hue='Spindle',data=new[:1830],cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[1],scale='area',scale_hue=True,
                palette={"Manual": "#20c073", "Automated": "#2976bb"})
ax[1].set(xlim=(0,2000),
            ylabel='')
ax[1].set_xlabel('Time (Sec)',fontweight='bold')
ax[1].set_title('Short recordings',fontweight='bold')
lgd2=ax[1].legend(loc='best',prop={'size':18})
ax = fig.add_subplot(122)
ax=sns.violinplot(y='Subject_',x='Onset',hue='Spindle',data=new[1830:],cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax,scale='area',scale_hue=True,
                palette={"Manual": "#20c073", "Automated": "#2976bb"})
ax.set(xlim=(0,2000),
            ylabel='')
ax.set_xlabel('Time (Sec)',fontweight='bold')
ax.set_title('Short recordings',fontweight='bold')
lgd2=ax.legend(loc='best',prop={'size':18})
fig.tight_layout()       
fig.savefig('manu vs auto(full).png',dpi=300)   
    
        
        
#manual_only = pd.concat(manual_only)
order=['Subject 5', 'Subject 6', 'Subject 8', 'Subject 9','Subject 10']
yticklabels = []
for y in order:
    temp = reset_yticklabels(y)
    yticklabels.append(temp)
f, ax = plt.subplots(figsize=(15,20),nrows=2,gridspec_kw = {'height_ratios':[5, 17]})        
new = pd.concat(manual_only['new'])
old = pd.concat(manual_only['old'])
new['Sub_']=new['Sub'].apply(reset_yticklabels)
old['Sub_']=old['Sub'].apply(reset_yticklabels)
ax[0]=sns.violinplot(y='Sub',x='Onset',hue='day',data=old,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[0],scale='area',scale_hue=True,
                    palette={"day1": "#2976bb", "day2": "#20c073"},order=order,)
ax[0].set(xlim=(0,4000),xlabel='',ylabel='Subject',yticklabels=yticklabels)
ax[0].set_title('Long recordings',fontweight='bold')  
lgd1=ax[0].legend(loc='upper right',prop={'size':28})  
yticklabels = []
for y in pd.unique(new['Sub']):
    yticklabels.append(reset_yticklabels(y))
ax[1]=sns.violinplot(y='Sub',x='Onset',hue='day',data=new,cut=0,split=True,
                  gridsize=20,inner="quart",ax=ax[1],scale='area',scale_hue=True,
                    palette={"day1": "#2976bb", "day2": "#20c073"})  
ax[1].set(xlim=(0,2000),ylabel='Subject',yticklabels=yticklabels)
ax[1].set_xlabel('Time (Sec)',fontweight='bold')
handles, labels = ax[1].get_legend_handles_labels()
ax[1].set_title('Short recordings',fontweight='bold')
lgd2=ax[1].legend(handles[::-1], labels[::-1],loc='upper left',prop={'size':28})
fig.tight_layout()
f.savefig('manu only(full).png',bbox_inches = 'tight',dpi=300)      
        
subjects=[11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,32] # missing 12, 26,27, 3
slow = 10,12   
slow_count=[]
fast = 12.5,14.5
fast_count=[]    
ll = 0.4; hh = 3.4
for ii in subjects:
    try:
        raw_file = [file for file in list_file_to_read if (str(ii) in file) and ('l2' in file) and ('fif' in file)]
        print(raw_file[0])
        raw = mne.io.read_raw_fif(raw_file[0],preload=True)
        raw.pick_channels(channelList)
        day = raw_file[0].split('_')[-1][:4]
        raw.filter(slow[0],slow[1])
        anno_name =[file for file in annotation_in_fold if (str(ii) in file) and (day in file) ]
        annotation=pd.read_csv(anno_name[0])
        time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
                                                                                                    channelList,annotation,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=ll,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=hh,
                                                                                                    front=300,back=100,sleep_stage=True,)
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
        annotation=pd.read_csv(anno_name[0])
        time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
                                                                                                    channelList,annotation,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=ll,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=hh,
                                                                                                    front=300,back=100,sleep_stage=True,)
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
        result = result[result.Onset > front]    
        fast_count.append([ii,len(result['Onset']),len(result['Onset'])/(raw.last_samp/raw.info['sfreq'] - total)])
    except:
        pass        
        
        
        
fast_count = pd.DataFrame(fast_count,columns=['subject','fast spindle count','fast spindle density'])        
slow_count = pd.DataFrame(slow_count,columns=['subject','slow spindle count','slow spindle density'])
fast_count.to_csv('fast_spindle_info(load2).csv')
slow_count.to_csv('slow_spindle_info(load2).csv')


subjects=[11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,32] # missing 12, 26,27, 3
slow = 10,12   
slow_count=[]
fast = 12.5,14.5
fast_count=[]    
ll = 0.4; hh = 3.4
for ii in subjects:
    try:
        raw_file = [file for file in list_file_to_read if (str(ii) in file) and ('l5' in file) and ('fif' in file)]
        print(raw_file[0])
        raw = mne.io.read_raw_fif(raw_file[0],preload=True)
        raw.pick_channels(channelList)
        day = raw_file[0].split('_')[-1][:4]
        raw.filter(slow[0],slow[1])
        anno_name =[file for file in annotation_in_fold if (str(ii) in file) and (day in file) ]
        annotation=pd.read_csv(anno_name[0])
        time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
                                                                                                    channelList,annotation,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=ll,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=hh,
                                                                                                    front=300,back=100,sleep_stage=True,)
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
        result = result[result.Onset > front]    
        slow_count.append([ii,len(result['Onset']),len(result['Onset'])/(raw.last_samp/raw.info['sfreq'] - total)]) 
    except:
        pass
for ii in subjects:
    try:
        raw_file = [file for file in list_file_to_read if (str(ii) in file) and ('l5' in file) and ('fif' in file)]
        print(raw_file[0])
        raw = mne.io.read_raw_fif(raw_file[0],preload=True)
        raw.pick_channels(channelList)
        day = raw_file[0].split('_')[-1][:4]
        raw.filter(fast[0],fast[1])
        anno_name =[file for file in annotation_in_fold if (str(ii) in file) and (day in file) ]
        annotation=pd.read_csv(anno_name[0])
        time_find,mean_peak_power,Duration,mph,mpl,auto_proba,auto_label=eegPinelineDesign.thresholding_filterbased_spindle_searching(raw,
                                                                                                    channelList,annotation,
                                                                                                    moving_window_size=windowSize,
                                                                                                    lower_threshold=ll,
                                                                                                    syn_channels=syn_channel,l_bound=0.5,
                                                                                                    h_bound=2,tol=0.5,higher_threshold=hh,
                                                                                                    front=300,back=100,sleep_stage=True,)
        result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
        result['Annotation'] = 'auto spindle'
        result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - back)]
        result = result[result.Onset > front]    
        fast_count.append([ii,len(result['Onset']),len(result['Onset'])/(raw.last_samp/raw.info['sfreq'] - total)])
    except:
        pass        
        
        
        
fast_count = pd.DataFrame(fast_count,columns=['subject','fast spindle count','fast spindle density'])        
slow_count = pd.DataFrame(slow_count,columns=['subject','slow spindle count','slow spindle density'])
fast_count.to_csv('fast_spindle_info(load5).csv')
slow_count.to_csv('slow_spindle_info(load5).csv')
'''
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
'''
