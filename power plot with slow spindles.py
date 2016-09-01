# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 11:46:55 2016

@author: install
"""

import eegPinelineDesign
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")
import pickle
import mne
import pandas as pd
#import scipy.signal as signal

def read_pickle(fileName):
    result_name = fileName[:-4] +'.p'
    pkl_file=open(result_name,'rb')
    result=pickle.load(pkl_file)
    pkl_file.close()
    return result

folderList = eegPinelineDesign.change_file_directory('D:\\NING - spindle')
subjectList = np.concatenate((np.arange(11,24),np.arange(25,31),np.arange(32,33)))
#subjectList = np.concatenate((np.arange(11,12))
#for idx in subjectList:
for idx in subjectList: # manually change the range, the second number is the position after the last stop
    
    folder_ = [folder_to_look for folder_to_look in folderList if (str(idx) in folder_to_look) and ('suj' in folder_to_look)]
    current_working_folder = eegPinelineDesign.change_file_directory('D:\\NING - spindle\\'+str(folder_[0]))
    list_file_to_read = [files for files in current_working_folder if ('vhdr' in files) and ('nap' in files)]
    for file_to_read in list_file_to_read:    
        raw=mne.io.read_raw_fif(file_to_read[:-5]+'.fif',preload=True,add_eeg_ref=False)
        epoch_length=10 #10sec
        epochs = eegPinelineDesign.make_overlap_windows(raw,epoch_length)
        epochs = np.unique(epochs)
        if idx == 26:
            channelList=['O1']
        else:
            channelList=['Cz']
        raw.pick_channels(channelList)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,stim=False)
        result={}
        alpha_C,DT_C,ASI,activity,ave_activity,psd_delta1,psd_delta2,psd_theta,psd_alpha,psd_beta,psd_gamma,slow_spindle,fast_spindle,slow_range,fast_range = eegPinelineDesign.epoch_activity(raw,picks=picks)
        activity = np.array(activity)
        ave_activity=np.array(ave_activity)
        alpha_C=np.array(alpha_C)
        DT_C=np.array(DT_C)
        ASI = np.array(ASI)
        
        range_alpha = np.array(psd_alpha).mean(1).max() - np.array(psd_alpha).mean(1).min()
        range_beta = np.array(psd_beta).mean(1).max() - np.array(psd_beta).mean(1).min()
        power_slow_spindle = np.array(slow_spindle)
        #chagne_slow_spindle
        range_slow_spindle = np.array(slow_spindle).mean(1).max() - np.array(slow_spindle).mean(1).min()
        My_ASI = (np.log2(np.array(psd_alpha).mean(1)/range_alpha) + np.log2(np.array(psd_beta).mean(1)/range_beta )) / np.log2(power_slow_spindle.mean(1)/range_slow_spindle)
        
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
        
        result_name = file_to_read[:-5] + '_slow_spindle.p'
        pickle.dump( result, open( result_name, "wb" ) )
        
        raw.filter(12,14)
        segment,time=raw[0,:]
            
        spindles = pd.read_csv(file_to_read[:-5]+'_slow_spindle.csv')
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
        ax1.plot(epochs[1:-1],np.array(psd_alpha),alpha=0.2)
        ax1.plot(epochs[1:-1],np.array(psd_alpha).mean(1),alpha=1.,color='black',label='average alpha activity')
        try:
            ax1.scatter(spindles['Onset'],np.mean(psd_alpha)*np.ones(len(spindles)))
        except:
            pass
        ax1.set(ylabel='power %s'%channelList,title='average alpha activity 8-12 Hz',xlim=[raw.first_samp/raw.info['sfreq'],raw.last_samp/raw.info['sfreq']])
        # fig 2
        ax2.plot(epochs[1:-1],psd_beta,alpha=0.2);ax2.set(title='average beta activity 12-20 Hz')
        ax2.plot(epochs[1:-1],np.array(psd_beta).mean(1),alpha=1.,color='black',label='beta')
        try:
            ax2.scatter(spindles['Onset'],np.mean(psd_beta)*np.ones(len(spindles)))
        except:
            pass
        # fig 3
        ax3.plot(time,segment[0,:],alpha=0.2)
        try:
            ax3.scatter(spindles['Onset'],0*np.ones(len(spindles))) ;ax3.set(title='%s band pass 12-14 Hz' % channelList[0],ylabel='$\mu$V')
        except:
            pass
        #fig 4
        ax4.plot(epochs[1:-1],pd.qcut(np.array(psd_alpha).mean(1),4,labels=False),label='categorical alpha',color='red') ;
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
        ax7.plot(epochs[1:-1],np.array(psd_beta).mean(1),'r',alpha=.4,label='beta')
        ax7.plot(epochs[1:-1],np.array(psd_alpha).mean(1),'b',alpha=.4,label='alpha');
        try:
            ax7.scatter(spindles['Onset'],np.mean(psd_beta)*np.ones(len(spindles)))
        except:
            pass
        
        ax7.set(title='mixed',xlim=[raw.first_samp/1000,raw.last_samp/1000],ylabel='power')
        ax7.legend();
        #fig 8 last two
        ax8_im.imshow(np.flipud(ave_activity.T),cmap=plt.cm.Blues,aspect='auto');
        ax8_im.set(yticks=np.arange(ave_activity.T.shape[0]),
                   yticklabels=(['gamma','beta','alpha','theta','delta2','delta1']))
        
        #fig 9 in middle,4th
        ax9.plot(epochs[1:-1],My_ASI,color='black',alpha=1.,label='ASI');ax9.set(title='$alpha + beta / fast_spindle_activity$',ylabel='ratio unit')
        ax99.plot(epochs[1:-1],pd.qcut(np.array(My_ASI),4,labels=False),color='r',alpha=0.3,label='cate ASI')
        ax99.set(yticks=[0,1,2,3],yticklabels=[1,2,3,4],ylim=[-0.1,3.1])
        try:
            ax9.scatter(spindles['Onset'],My_ASI.mean()*np.ones(len(spindles)))
        except:
            pass
        ax9.set(xlim=[raw.first_samp/1000,raw.last_samp/1000])
        ax9.legend();ax99.legend()
        #fig 10 last
        ax_im=plt.subplot(616,sharex=ax8_im)
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
        pic_fileName = fileName[:-4] + '_slow_spindle.png'
        plt.savefig(pic_fileName)