# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:53:51 2016

@author: asus-task
"""

import eegPinelineDesign
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import mne
import seaborn as sns
""" click on the last botton under the console line, which is 'option',
    select 'open a new python console'
    run this script in that new concole
"""



folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33)))
result=dict(subject=[],spindle_RMS_amplitude_mean=[],spindle_density=[],spindle_measurement=[],length=[],
            sampleF=[],lowpass=[],highpass=[],Duration_mean=[],spindle_RMS_amplitude_std=[],
            Duration_std=[],spindle_count=[],condition=[])
keyword = 'slow' # manually change to slow/fast
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if str(idx) in folder_to_look]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('csv' in files) and ('nap' in files) and (keyword in files)]
    print(list_file_to_read)
    for file_to_read in list_file_to_read:
        raw_file_to_read = file_to_read[:-17] + '.vhdr'
        raw=mne.io.read_raw_brainvision(raw_file_to_read,scale=1e6,preload=False)
        subject_nap_time = raw.last_samp/raw.info['sfreq'] - 200
        
        spindleFile = pd.read_csv(file_to_read,sep=',')
        spindle_count=len(spindleFile)
        spindle_RMS_amplitude_mean = spindleFile['Amplitude'].mean()
        spindle_RMS_amplitude_std = spindleFile['Amplitude'].std()
        spindle_density= len(spindleFile)/subject_nap_time
        duration_mean = spindleFile['Duration'].mean()
        duration_std = spindleFile['Duration'].std()
        spindle_measurement = spindle_density * spindleFile['Amplitude'].mean() 
        condition='load %d'%int(file_to_read.split('_')[1][1])
        
        
        result['subject'].append(file_to_read[:-4])
        result['spindle_RMS_amplitude_mean'].append(spindle_RMS_amplitude_mean)
        result['spindle_RMS_amplitude_std'].append(spindle_RMS_amplitude_std)
        result['spindle_density'].append(spindle_density)
        result['spindle_measurement'].append(spindle_measurement)
        result['length'].append(subject_nap_time)
        result['sampleF'].append(raw.info['sfreq'])
        result['lowpass'].append(raw.info['lowpass'])
        result['highpass'].append(raw.info['highpass'])
        result['Duration_mean'].append(duration_mean)
        result['Duration_std'].append(duration_std)
        result['spindle_count'].append(spindle_count)
        result['condition'].append(condition)

result = pd.DataFrame(result)
result['length'] = result['length']/60
result['spindle_density']=result['spindle_density'] * 60
result.rename(index=result['subject'],columns={'spindle_RMS_amplitude_mean':'spindle mean amplitude (RMS amplitude)',
              'spindle_density':'spindle density (# of spindles per minute)',
              'length':'length of time used (min)','sampleF':'sampling frequency',
              'Duration_mean':'Mean duration of spindles',
              'spindle_measurement':'spindle measurement = density * amplitue'},inplace=True)
result = result.drop('subject',1)
########### here is where you can find the file #######################
_=eegPinelineDesign.change_file_directory('D:\\NING - spindle')
#######################################################################
result.to_csv('spindle result_%s.csv' % keyword,index=True)

# visualizing
lm=sns.lmplot(x='spindle density (# of spindles per minute)',y='spindle mean amplitude (RMS amplitude)',data=result,hue='condition',markers=["o", "x"],robust=True)
lm.set(title='Robust regression fit between %s spindle density and %s spindle amplitude'%(keyword,keyword))
lm.savefig('%s spindle robust regression fit.png'%keyword)
g=sns.jointplot(x='spindle density (# of spindles per minute)',y='spindle mean amplitude (RMS amplitude)',data=result,kind="reg",annot_kws=dict(stat="r"))
g.savefig('%s spindle joint plot.png'%keyword)
result.hist()