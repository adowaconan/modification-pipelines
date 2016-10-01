# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:52:52 2016

@author: install
"""

import eegPinelineDesign
import pandas as pd
import numpy as np
import mne
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
chan_dict={'Ch56': 'TP8', 'Ch61': 'F6', 'Ch3': 'F3', 'Ch45': 'P1', 'Ch14': 'P3', 
           'Ch41': 'C1', 'Ch1': 'Fp1', 'Ch46': 'P5', 'Ch7': 'FC1', 'Ch37': 'F5', 
           'Ch21': 'TP10', 'Ch8': 'C3', 'Ch11': 'CP5', 'Ch28': 'FC6', 'Ch17': 'Oz', 
           'Ch39': 'FC3', 'Ch38': 'FT7', 'Ch58': 'C2', 'Ch33': 'AF7', 'Ch48': 'PO3', 
           'Ch9': 'T7', 'Ch49': 'POz', 'Ch2': 'Fz', 'Ch15': 'P7', 'Ch20': 'P8', 
           'Ch60': 'FT8', 'Ch57': 'C6', 'Ch32': 'Fp2', 'Ch29': 'FC2', 'Ch59': 'FC4', 
           'Ch35': 'AFz', 'Ch44': 'CP3', 'Ch47': 'PO7', 'Ch30': 'F4', 'Ch62': 'F2', 
           'Ch4': 'F7', 'Ch24': 'Cz', 'Ch31': 'F8', 'Ch64': 'ROc', 'Ch23': 'CP2', 
           'Ch25': 'C4', 'Ch40': 'FCz', 'Ch53': 'P2', 'Ch19': 'P4', 'Ch27': 'FT10', 
           'Ch50': 'PO4', 'Ch18': 'O2', 'Ch55': 'CP4', 'Ch6': 'FC5', 'Ch12': 'CP1', 
           'Ch16': 'O1', 'Ch52': 'P6', 'Ch5': 'FT9', 'Ch42': 'C5', 'Ch36': 'F1', 
           'Ch26': 'T8', 'Ch51': 'PO8', 'Ch34': 'AF3', 'Ch22': 'CP6', 'Ch54': 'CPz', 
           'Ch13': 'Pz', 'Ch63': 'LOc', 'Ch43': 'TP7'}
def rescale(M):
    M = np.array(M)
    return (M - M.min())/(M.max() - M.min())+2
def PSDW(a,b,c):
    return ((a+b)/c)
had = True
current_working_folder=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set\\')
chanName=['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'Pz',
          'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 
          'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 
          'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 
          'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 
          'F2', 'LOc', 'ROc', 'Aux1', 'STI 014']
"""
This script is to correct old data using ICA
"""
subjectList=[5,6,8,9,10]
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files) and ('suj%d'%idx in files)]
    print(list_file_to_read);low=1;high=200
    for file_to_read in list_file_to_read:
        if had:
            raw = mne.io.read_raw_brainvision('D:\\NING - spindle\\training set\\'+file_to_read,preload=True,scale=1e6)#[:-5] + '.fif',add_eeg_ref=False
        else:
            try:
                raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=180,eogReject=300,n_ch=-2)
            except:
                try:
                    raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=240,eogReject=300,n_ch=-2)
                except:
                    try:
                        raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=300,eogReject=300,n_ch=-2)
                    except:
                        raw = eegPinelineDesign.load_data('D:\\NING - spindle\\training set\\'+file_to_read,low_frequency=1,high_frequency=50,eegReject=360,eogReject=300,n_ch=-2)
            raw.save(file_to_read[:-5] + '.fif',overwrite=True)
        try:
            raw.rename_channels(chan_dict)
        except:
            pass
        
        epoch_length=10 #10sec
        channelList=['Cz']
        raw.pick_channels(channelList)
        raw.filter(1,40)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,stim=False)
        result={}
        alpha_C,DT_C,ASI,activity,ave_activity,psd_delta1,psd_delta2,psd_theta,psd_alpha,psd_beta,psd_gamma,slow_spindle,fast_spindle,slow_range,fast_range,epochs = eegPinelineDesign.epoch_activity(raw,picks=picks)
        epochs=np.unique(epochs)        
        activity = np.array(activity)
        ave_activity=np.array(ave_activity)
        alpha_C=np.array(alpha_C)
        DT_C=np.array(DT_C)
        ASI = np.array(ASI)
        power_fast_spindle = np.array(fast_spindle)
        My_ASI = PSDW(rescale(np.array(psd_alpha).mean(1)),rescale(np.array(psd_beta).mean(1)) ,rescale(power_fast_spindle.mean(1)))
        
        result['alpha activity']=alpha_C;result['sum of delta and theta']=DT_C
        result['activity across 6 bands']=activity;result['delta 0-2']=psd_delta1
        result['delta 2-4']=psd_delta2;result['theta']=psd_theta;result['alpha']=psd_alpha
        result['beta']=psd_beta;result['gamma']=psd_gamma
        result['beta_mean']=np.array(psd_beta).mean(1)
        result['gamma_std']=np.array(psd_gamma).std(1)
        result['activity']=np.array(ave_activity)
        result['slow']=np.array(slow_spindle)
        result['fast']=np.array(fast_spindle)
        result['my ASI']=np.array(My_ASI)
        
        result_name = file_to_read[:-5] + '_fast_spindle.p'
        pickle.dump( result, open( result_name, "wb" ) )
        
        raw.filter(12.5,14.5)
        segment,time=raw[0,:]
            
        spindles = pd.read_csv(file_to_read[:-5]+'_fast_spindle.csv')
        syn_gamma = np.array(psd_gamma).std(1)
        mean_beta = np.array(psd_beta).mean(1)
        
        fig=plt.figure(figsize=(40,40))
        fig.suptitle(file_to_read[:-5])
        ax1 = plt.subplot(631)
        ax2 = plt.subplot(632,sharex=ax1)
        ax3 = plt.subplot(633,sharex=ax1)
        ax4 = plt.subplot(634,sharex=ax1)
        ax5 = plt.subplot(635,sharex=ax1)
        ax6 = plt.subplot(636,sharex=ax1)
        ax7 = plt.subplot(613);
        ax8_im = plt.subplot(615)
        ax9 = plt.subplot(614,sharex=ax7);ax99=ax9.twinx()
        # fig 1
        ax1.plot(epochs[1:-1],np.array(psd_alpha),alpha=0.1)
        ax1.plot(epochs[1:-1],np.array(psd_alpha).mean(1),alpha=1.,color='black',label='average alpha activity')
        try:
            ax1.scatter(spindles['Onset'],np.mean(psd_alpha)*np.ones(len(spindles)))
        except:
            pass
        ax1.set(ylabel='power %s'%channelList,title='average alpha activity 8-12 Hz',xlim=[raw.first_samp/raw.info['sfreq'],raw.last_samp/raw.info['sfreq']])
        # fig 2
        ax2.plot(epochs[1:-1],psd_beta,alpha=0.1);ax2.set(title='average beta activity 12-20 Hz')
        ax2.plot(epochs[1:-1],np.array(psd_beta).mean(1),alpha=1.,color='black',label='beta')
        try:
            ax2.scatter(spindles['Onset'],np.mean(psd_beta)*np.ones(len(spindles)))
        except:
            pass
        # fig 3
        ax3.plot(time,segment[0,:],alpha=0.2)
        try:
            ax3.scatter(spindles['Onset'],0*np.ones(len(spindles))) ;
        except:
            pass
        ax3.set(title='%s %s band pass 12-14 Hz' % (file_to_read[:-5],channelList[0]),ylabel='$\mu$V')
        #fig 4
        ax4.plot(epochs[1:-1],pd.cut(np.array(psd_alpha).mean(1),4,labels=False),label='categorical alpha',color='red') ;
        try:
            ax4.scatter(spindles['Onset'],1.5*np.ones(len(spindles)),color='blue',marker='s')
        except:
            pass    
        ax4.set(title='alpha in categories',yticks=[0,1,2,3],yticklabels=[1,2,3,4],ylabel='categories')
        #fig 5
        ax5.plot(epochs[1:-1],np.array(slow_spindle),alpha=0.2);ax5.set(title='slow spindle %.2f - %.2f Hz' % (slow_range.min(),slow_range.max()))
        ax5.plot(epochs[1:-1],np.array(slow_spindle).mean(1),color='black',alpha=1.)
        try:
            ax5.scatter(spindles['Onset'],np.mean(slow_spindle)*np.ones(len(spindles)))
        except:
            pass
        #fig 6
        ax6.plot(epochs[1:-1],np.array(fast_spindle),alpha=0.2);ax6.set(title='fast spindle %.2f - %.2f Hz' % (fast_range.min(),fast_range.max()))
        ax6.plot(epochs[1:-1],np.array(fast_spindle).mean(1),color='black',alpha=1.)
        try:
            ax6.scatter(spindles['Onset'],np.mean(fast_spindle)*np.ones(len(spindles)))
        except:
            pass
        #fig 7 3rd in middle
        #ax7.plot(epochs[1:-1],np.array(psd_beta).mean(1),'r',alpha=.4,label='beta')
        #ax7.plot(epochs[1:-1],np.array(psd_alpha).mean(1),'b',alpha=.4,label='alpha');
        csv_to_read = [files for files in current_working_folder if ('txt' in files) and (file_to_read.split('_')[1][-1] in files) and ('%d'%idx in files)]
        annotation = pd.read_csv(csv_to_read[0],index_col=None)  
        annotation = annotation[['Onset','Annotation']]
        key=re.compile('Markon',re.IGNORECASE)
        temp=[]
        for row in enumerate(annotation.iterrows()):
            #print(key.search(row[1][-1][-1]))
            if key.search(row[1][-1][-1]):
                temp.append([row[1][-1][-2],row[1][-1][-1]])
        temp = pd.DataFrame(temp,columns=['Onset','Annotation'])
        
        temp['Annotation']=temp.Annotation.apply(eegPinelineDesign.recode_annotation)
        temp.plot(x='Onset',y='Annotation',style='.-',ax=ax7)
        try:
            ax7.scatter(spindles['Onset'],np.mean(np.arange(len(pd.unique(temp.Annotation.values))))*np.ones(len(spindles)))
        except:
            pass
        
        ax7.set(title='sleep stages',ylabel='stage',
                yticks=np.arange(len(pd.unique(temp.Annotation.values))),
                yticklabels=['w','1','2','3'])
        #ax7.legend();
        #fig 8 last two
        ax8_im.imshow(np.flipud(ave_activity.T),cmap=plt.cm.Blues,aspect='auto');
        ax8_im.set(yticks=np.arange(ave_activity.T.shape[0]),
                   yticklabels=(['gamma','beta','alpha','theta','delta2','delta1']))
        
        #fig 9 in middle,4th
        ax9.plot(epochs[1:-1],My_ASI,color='black',alpha=1.,label='ASI');ax9.set(title='alpha + beta / fast_spindle_activity',ylabel='ratio unit')
        ax99.plot(epochs[1:-1],pd.cut(np.array(My_ASI),4,labels=False),color='r',alpha=0.3,label='cate ASI')
        ax99.set(yticks=np.arange(4),yticklabels=np.arange(4)+1,ylim=[-0.1,3.1])
        try:
            ax9.scatter(spindles['Onset'],My_ASI.mean()*np.ones(len(spindles)))
        except:
            pass
        ax9.set(xlim=[raw.first_samp/1000,raw.last_samp/1000])
        ax9.legend();ax99.legend()
        #fig 10 last
        ax_im=fig.add_subplot(616,sharex=ax8_im)
        xx,yy = np.meshgrid(epochs[1:-1],np.arange(activity.shape[1]))
        im=ax_im.imshow(np.flipud(activity.T),cmap=plt.cm.Blues,aspect='auto')
        ax_im.set(yticks=activity.T.shape[0]-np.arange(activity.T.shape[0])[[10,30,65,90,150]],
                  yticklabels=(['delta1','delta2','theta','alpha','beta']),
                               title='mean power spectral density',
                               xticks=np.arange(len(epochs[1:-1]))[0::20],
                               xticklabels=epochs[1:-1:20],
                               xlabel='time (sec)',ylabel='frequency bands')
        #plt.colorbar(im)
        plt.tight_layout()
        fileName = file_to_read[:-5] + '.csv'
        pic_fileName = fileName[:-4] + '_fast_spindle.png'
        fig.savefig(pic_fileName)