# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:35:44 2016

@author: ning
"""

import eegPinelineDesign
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

try:
    eegPinelineDesign.change_file_directory('D:\\NING - spindle\\suj29')
except:
    pass

file_to_read = 'suj29_l5nap_day1.fif'
valiF = 'suj29_nap_day1_edited_annotations.txt'
peak_time,result,spindles, match, mismatch = \
eegPinelineDesign.EEGpipeline_by_epoch(file_to_read,validation_file=valiF,lowCut=12.5,highCut=14.5,majority=5,mul=0.9)
plt.close('all')
"""
file1 = pd.read_csv('suj13_l2nap_day2',sep=',')
file2 = pd.read_csv('suj13_nap_day2_annotations.txt',sep=',')
fileName = 'suj13_l2nap_day2'


labelFind = eegPinelineDesign.re.compile('spindle',eegPinelineDesign.re.IGNORECASE)
spindles=[]# take existed annotations
for row in file2.iterrows():
    currentEvent = row[1][-1]
    if labelFind.search(currentEvent):
        spindles.append(row[1][0])# time of marker    
spindles = np.array(spindles)

peak_time = file1['0'].values
Time_found = peak_time
match=[]
mismatch=[]
for item in Time_found:
    if any(abs(item - spindles)<1):
        match.append(item)
    else:
        mismatch.append(item)
fig,ax=plt.subplots()
ax.bar(np.arange(3),[len(match),len(mismatch),len(spindles)],align="center")
ax.text(0-0.4,len(match)+5,'match rate is %.2f' % (len(match)/len(spindles)))
ax.text(0-0.4,len(match)+10,'auto detection found %d' % (len(Time_found)))
ax.set_title(fileName+' before constraint by sleeping stage')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['match','mismatch','man marked spindles'])



labelFind = eegPinelineDesign.re.compile('Marker: Markon: 2',eegPinelineDesign.re.IGNORECASE)
stage2=[]# take existed annotations
for row in file2.iterrows():
    currentEvent = row[1][-1]
    if labelFind.search(currentEvent):
        stage2.append([row[1][0],row[1][0]+30])# time of marker    
stage2 = np.array(stage2)

Time_=[]
for item in Time_found:
    if any(list(eegPinelineDesign.intervalCheck(stages,item) for stages in stage2)):
        Time_.append(item)
        
match=[]
match_=[]
mismatch=[]
for item in Time_:
    if any(abs(item - spindles)<2):
        match.append(item)
        match_.append(min(enumerate(spindles), key=lambda x: abs(x[1]-item))[1])
    else:
        mismatch.append(item)
fig,ax=plt.subplots()
ax.bar(np.arange(3),[len(match),len(mismatch),len(spindles)],align="center")
ax.text(0-0.4,len(match)+5,'match rate is %.2f' % (len(match)/len(spindles)))
ax.text(0-0.4,len(match)+10,'auto detection found %d' % (len(Time_)))
ax.text(2-0.4,len(spindles)+10,'spindles marked %d' % (len(spindles)))
ax.set_title(fileName+' after constraint by sleeping stage')
ax.set_xticks(np.arange(3))
ax.set_xticklabels(['match','mismatch','man marked spindles'])



try:
    eegPinelineDesign.change_file_directory('C:/Users/ning/Downloads/training set')
except:
    pass
EDFfiles, Annotationfiles = eegPinelineDesign.split_type_of_files()

file_to_read='suj13_l2nap_day2.vhdr'
channelList = ['F3','F4','C3','C4','O1','O2','LOc', 'ROc']
raw = eegPinelineDesign.load_data(file_to_read,channelList,12,14)
# select channels of interests
channelList = ['F3','F4','C3','C4','O1','O2']
raw.pick_channels(channelList)
#raw_narrow.plot_psd(fmax=30,tmin=200,tmax=2000)

#eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\training set\\figure\\match')
for item in match:
    fig=plt.figure(figsize=(15,15))
    fig.suptitle(item)
    for ii, names in enumerate(channelList):
        segment, time = eegPinelineDesign.cut_segments(raw,item,ii,5)
        ax=plt.subplot(6,1,ii+1)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.plot(time,segment[0,:],label=names)
        ax.set_title('matched between auto and manual detection '+names,fontsize=20)
        ax.set_xlabel('Time');ax.set_ylabel('$\mu$V')
    plt.tight_layout()
    figureName=str(item)+'match_.png'
    #plt.savefig(figureName)
#eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\training set\\figure\\mismatch')
for item in mismatch:
    fig=plt.figure(figsize=(15,15))
    fig.suptitle(item)
    for ii, names in enumerate(channelList):
        segment, time = eegPinelineDesign.cut_segments(raw,item,ii,5)
        ax=plt.subplot(6,1,ii+1)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.plot(time,segment[0,:],label=names)
        ax.set_title('mismatched between auto and manual detection '+names,fontsize=20)
        ax.set_xlabel('Time');ax.set_ylabel('$\mu$V')
    plt.tight_layout()
    figureName=str(item)+'mismatch_.png'
    #plt.savefig(figureName)

#eegPinelineDesign.change_file_directory('C:\\Users\\ning\\Downloads\\training set\\figure\\not found')
for item in list(set(spindles) - set(match_) ):
    fig=plt.figure(figsize=(15,15))
    fig.suptitle(item)
    for ii, names in enumerate(channelList):
        segment, time = eegPinelineDesign.cut_segments(raw,item,ii,5)
        ax=plt.subplot(6,1,ii+1)
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.plot(time,segment[0,:],label=names)
        ax.set_title('spindles that is not detected by the auto algorithm '+names,fontsize=20)
        ax.set_xlabel('Time');ax.set_ylabel('$\mu$V')
    plt.tight_layout()
    figureName=str(item)+'not found_.png'
    #plt.savefig(figureName)
"""














