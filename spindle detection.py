import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hmean,trim_mean

def window_rms(a, window_size):
    """
    a: numpy array of sample data
    window_size: size of convolution kernel
    """
    a2 = np.power(a,2)# sqaure all data points
    window = scipy.signal.gaussian(window_size,(window_size/.68)/2)# apply gaussian window with length of window_size samples
    return np.sqrt(np.convolve(a2, window, 'same')/len(a2)) * 1e2 # convolve along entire data and return root mean sqaures with the same length of the sample data
def trimmed_std(data,percentile):
    temp=data.copy()
    temp.sort()
    percentile = percentile / 2
    low = int(percentile * len(temp))
    high = int((1. - percentile) * len(temp))
    return temp[low:high].std(ddof=0)

def get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,moving_window_size=200,threshold=.9,syn_channels=3,l_freq=0,h_freq=200,l_bound=0.5,h_bound=2,tol=1):
    """
    raw: data after preprocessing
    channelList: channel list of interest
    file_to_read: raw data file names
    moving_window_size: size of the moving window for convolved root mean square computation
    threshold: threshold for spindle detection: threshold = mean + threshold * std
    syn_channels: criteria for selecting spindles: at least # of channels have spindle instance and also in the mean channel
    l_freq: high pass frequency for spindle range
    h_freq: low pass frequency for spindle range
    l_bound: low boundary for duration of a spindle instance
    h_bound: high boundary for duration of a spindle instance
    tol : tolerance for determing spindles (criteria in time)
    """
    mul=threshold;nn=4.5
    
    time=np.linspace(0,raw.last_samp/raw.info['sfreq'],raw._data[0,:].shape[0])
    RMS = np.zeros((len(channelList),raw._data[0,:].shape[0]))
    peak_time={} #preallocate
    fig=plt.figure(figsize=(40,40))
    ax=plt.subplot(311)
    ax1=plt.subplot(312,sharex=ax)
    ax2=plt.subplot(313,sharex=ax)
    for ii, names in enumerate(channelList):

        peak_time[names]=[]
        segment,_ = raw[ii,:]
        RMS[ii,:] = window_rms(segment[0,:],moving_window_size) # window of 200ms
        mph = trim_mean(RMS[ii,100000:-30000],0.05) + mul * trimmed_std(RMS[ii,:],0.05) # higher sd = more strict criteria
        mpl = trim_mean(RMS[ii,100000:-30000],0.05) + nn * trimmed_std(RMS[ii,:],0.05)
        pass_= RMS[ii,:] > mph

        up = np.where(np.diff(pass_.astype(int))>0)
        down = np.where(np.diff(pass_.astype(int))<0)
        up = up[0]
        down = down[0]
        ###############################
        #print(down[0],up[0])
        if down[0] < up[0]:
            down = down[1:]
        #print(down[0],up[0])
        #############################
        if (up.shape > down.shape) or (up.shape < down.shape):
            size = np.min([up.shape,down.shape])
            up = up[:size]
            down = down[:size]
        C = np.vstack((up,down))
        for pairs in C.T:
            if l_bound < (time[pairs[1]] - time[pairs[0]]) < h_bound:
                TimePoint = np.mean([time[pairs[1]],time[pairs[0]]])
                SegmentForPeakSearching = RMS[ii,pairs[0]:pairs[1]]
                if np.max(SegmentForPeakSearching) < mpl:
                    temp_temp_time = time[pairs[0]:pairs[1]]
                    ints_temp = np.argmax(SegmentForPeakSearching)
                    peak_time[names].append(temp_temp_time[ints_temp])
                    ax.scatter(temp_temp_time[ints_temp],mph+0.1*mph,marker='s',
                               color='blue')
        ax.plot(time,RMS[ii,:],alpha=0.2,label=names)
        ax2.plot(time,segment[0,:],label=names,alpha=0.3)
        ax2.set(xlabel="time",ylabel="$\mu$V",xlim=(time[0],time[-1]),title=file_to_read[:-5]+' band pass %.1f - %.1f Hz' %(l_freq,h_freq))
        ax.set(xlabel="time",ylabel='RMS Amplitude',xlim=(time[0],time[-1]),title='auto detection on each channels')
        ax1.set(xlabel='time',ylabel='Amplitude')
        ax.axhline(mph,color='r',alpha=0.03)
        ax2.legend();ax.legend()

    peak_time['mean']=[];peak_at=[];duration=[]
    RMS_mean=hmean(RMS)
    ax1.plot(time,RMS_mean,color='k',alpha=0.3)
    mph = trim_mean(RMS_mean[100000:-30000],0.05) + mul * RMS_mean.std()
    mpl = trim_mean(RMS_mean[100000:-30000],0.05) + nn * RMS_mean.std()
    pass_ = RMS_mean > mph
    up = np.where(np.diff(pass_.astype(int))>0)
    down= np.where(np.diff(pass_.astype(int))<0)
    up = up[0]
    down = down[0]
    ###############################
    #print(down[0],up[0])
    if down[0] < up[0]:
        down = down[1:]
    #print(down[0],up[0])
    #############################
    if (up.shape > down.shape) or (up.shape < down.shape):
        size = np.min([up.shape,down.shape])
        up = up[:size]
        down = down[:size]
    C = np.vstack((up,down))
    for pairs in C.T:
        
        if 0.5 < (time[pairs[1]] - time[pairs[0]]) < 2:
            TimePoint = np.mean([time[pairs[1]] , time[pairs[0]]])
            SegmentForPeakSearching = RMS_mean[pairs[0]:pairs[1],]
            if np.max(SegmentForPeakSearching)< mpl:
                temp_time = time[pairs[0]:pairs[1]]
                ints_temp = np.argmax(SegmentForPeakSearching)
                peak_time['mean'].append(temp_time[ints_temp])
                peak_at.append(SegmentForPeakSearching[ints_temp])
                ax1.scatter(temp_time[ints_temp],mph+0.1*mph,marker='s',color='blue')
                duration_temp = time[pairs[1]] - time[pairs[0]]
                duration.append(duration_temp)
    ax1.axhline(mph,color='r',alpha=1.)
    ax1.set_xlim([time[0],time[-1]])


    time_find=[];mean_peak_power=[];Duration=[]
    for item,PEAK,duration_time in zip(peak_time['mean'],peak_at,duration):
        temp_timePoint=[]
        for ii, names in enumerate(channelList):
            try:
                temp_timePoint.append(min(enumerate(peak_time[names]), key=lambda x: abs(x[1]-item))[1])
            except:
                temp_timePoint.append(item + 2)
        try:
            if np.sum((abs(np.array(temp_timePoint) - item)<tol).astype(int))>syn_channels:
                time_find.append(float(item))
                mean_peak_power.append(PEAK)
                Duration.append(duration_time)
        except:
            pass
    return time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at

def save_spindles(time_find,mean_peak_power,Duration,file_to_read,fig):
    df = pd.DataFrame({'Onset':time_find,'RMS amplitude':mean_peak_power,"Duration":Duration})
    df.to_csv(file_to_read[:-5]+'.csv')

    fig.savefig(file_to_read[:-5]+'.png')


file_to_read = fileName
channelList = ['F3','F4','C3','C4','O1','O2']
time_find,mean_peak_power,Duration,fig,ax,ax1,ax2,peak_time,peak_at=get_Onest_Amplitude_Duration_of_spindles(raw,channelList,file_to_read,moving_window_size=200,threshold=.9,syn_channels=3,l_freq=0,h_freq=200,l_bound=0.5,h_bound=2,tol=1)
result = pd.DataFrame({"Onset":time_find,"Amplitude":mean_peak_power,'Duration':Duration})
result['Annotation'] = 'auto spindle'
result = result[result.Onset < (raw.last_samp/raw.info['sfreq'] - 100)]
result = result[result.Onset > 100]
Time_ = result.Onset.values
ax2.scatter(Time_,np.zeros(len(Time_)),marker='s',color='blue')
