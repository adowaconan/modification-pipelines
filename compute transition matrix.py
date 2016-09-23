# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 14:02:17 2016

@author: ning
"""

import numpy as np
from collections import Counter
import os
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\ning\\OneDrive\\python works\\modification-pipelines')
import eegPinelineDesign
results = pickle.load(open('sleep annotation.p','rb'))
fig=plt.figure(figsize=(30,15));cnt=1
for name in results.keys():
    sub=results[name]
    annotation=sub['annotation']
    key=re.compile('Marker',re.IGNORECASE)
    temp=[]
    for row in enumerate(annotation.iterrows()):
        #print(row[1][-1])
        if key.search(row[1][-1][-1]):
            temp.append([row[1][-1][-2],row[1][-1][-1]])
    temp = pd.DataFrame(temp,columns=['Onset','annotation'])
    
    temp['annotation']=temp.annotation.apply(eegPinelineDesign.recode_annotation)
    #ax=temp.plot(x='Onset',y='annotation',style='.-',yticks=[0,1,2,3],xticks=temp.Onset.values[::10])
    #_=ax.set(yticklabels=['w','1','2','SWS'],title=name,xlabel='time (sec)',ylabel='stage',
    #         xlim=(-2,1900))
    a=temp['annotation'].values
    state = np.vstack((a,a,a))
    state = state.T.reshape((3*len(a)))
    state = state[:-1]
    Nstate=len(np.unique(temp['annotation'].values))
    tranMat = np.zeros((Nstate,Nstate))
    for (x,y), c in Counter(zip(a, a[1:])).items():
        tranMat[x-1][y-1] = c
    tranMat = tranMat / tranMat.sum(1)
    print('transisition matrix\n\n',name,'\n\n', tranMat)
    ax1=fig.add_subplot(2,5,cnt)
    ax1.pcolor(tranMat,cmap=plt.cm.Blues)
    ax1.xaxis.tick_top()
    ax1.set(xticks=np.arange(Nstate)+0.5,yticks=np.arange(Nstate)+0.5,
            xticklabels=['w','1','2','SWS'],yticklabels=['w','1','2','SWS'])
    plt.title(name,y=1.08)
    for y in range(tranMat.shape[0]):
        for x in range(tranMat.shape[1]):
            ax1.annotate('%.4f'%tranMat[y,x],xy=(x+0.3,y+0.5))
    if cnt == 1:
        ax1.set(xlabel='stage',ylabel='stage')
    
    Ncat=4
    Obs = results[name]['result']['my ASI']
    Obs = pd.DataFrame({'Obs':pd.cut(Obs,Ncat,labels=False)})
    #ax=Obs.plot(style='.-',yticks=np.arange(Ncat))
    #_=ax.set(title=name,yticklabels=np.arange(Ncat)+1)
    EmissionMat = np.zeros((Nstate,Ncat))
    for (x,y),c in Counter(zip(state,Obs['Obs'].values)).items():
        EmissionMat[x-1][y-1] = c
    EmissionMat = EmissionMat / EmissionMat.sum(axis=1, keepdims=True)
    print('emissin matrix\n\n',name,'\n\n',EmissionMat)
    ax2=fig.add_subplot(2,5,cnt+5)
    ax2.pcolor(EmissionMat,cmap=plt.cm.Blues)
    ax2.xaxis.tick_top()
    ax2.set(yticks=np.arange(Nstate)+0.5,xticks=np.arange(Ncat)+0.5,
            yticklabels=['w','1','2','SWS'],xticklabels=np.arange(Ncat)+1)
    for y in range(EmissionMat.shape[0]):
        for x in range(EmissionMat.shape[1]):
            ax2.annotate('%.4f'%EmissionMat[y,x],xy=(x+0.3,y+0.5))
    if cnt == 1:
        ax2.set(xlabel='observation',ylabel='stage')
    cnt +=1
fig.savefig('transition matrix and emission matrix.png')