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
    if Nstate == 4:
        yticklabels=['SWS','2','1','w']
    else:
        yticklabels=['2','1','w']
    tranMat = np.zeros((Nstate,Nstate))
    for (x,y), c in Counter(zip(a, a[1:])).items():
        tranMat[x-1][y-1] = c
    tranMat = tranMat / tranMat.sum(axis=1, keepdims=True)
    print('transisition matrix\n\n',name,'\n\n', tranMat)
    ax1=fig.add_subplot(2,5,cnt)
    ax1.pcolor(np.flipud(tranMat),cmap=plt.cm.Blues)
    ax1.xaxis.tick_top()
    ax1.set(xticks=np.arange(Nstate)+0.5,yticks=np.arange(Nstate)+0.5,
            xticklabels=['w','1','2','SWS'],yticklabels=yticklabels)
    plt.title(name,y=1.08)
    for x in range(tranMat.shape[0]):
        for y in range(tranMat.shape[1]):
            #print(np.flipud(tranMat)[x,y])
            ax1.annotate('%.4f'%np.fliplr(tranMat.T)[x,y],xy=(x+0.3,y+0.5))
    if cnt == 1:
        ax1.set(xlabel='stage',ylabel='stage')
    
    Ncat=5
    global_=pd.read_csv('global ASI.csv')
    _,bins=pd.cut(global_['global ASI'],Ncat,labels=False,retbins=True)
    Obs = results[name]['result']['my ASI']
    Obs = pd.DataFrame({'Obs':pd.cut(Obs,bins,labels=False)})
    max_=Obs['Obs'].max()
    #Obs['Obs']=Obs['Obs'].apply(lambda x:  max_-x)
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
            yticklabels=yticklabels,xticklabels=np.arange(Ncat)+1)
    for x in range(EmissionMat.shape[0]):
        for y in range(EmissionMat.shape[1]):
            ax2.annotate('%.4f'%EmissionMat[x,y],xy=(y+0.1,x+0.5))
    if cnt == 1:
        ax2.set(xlabel='observation',ylabel='stage')
    cnt +=1
#fig.savefig('transition matrix and emission matrix.png')
    
    from hmmlearn import hmm
    seq = np.vstack([Obs['Obs'].values[:state.shape[0]],state])
    seq = seq.T
    model = hmm.GaussianHMM(n_components=tranMat.shape[0],random_state=0,n_iter=100,tol=1e-5,covariance_type='diag',
                            params='tmc',init_params='st')
    model.transmat_=tranMat
    #model.startprob_prior=np.concatenate(([1],np.zeros(tranMat.shape[0]-1)))
    model.startprob_=np.concatenate(([1],np.zeros(tranMat.shape[0]-1)))
    model.fit(seq)
    model.predict(seq)
    print(model.decode(seq)[1],seq[:,1],1- np.nonzero(model.decode(seq)[1] - seq[:,1])[0].shape[0]/len(seq),1/tranMat.shape[0])
    #print(model.decode(seq))
    
#from sklearn.cross_validation import train_test_split
#from sklearn.linear_model import LinearRegression
#t=results[name]['result']['my ASI'][:state.shape[0]];
#X = np.vstack([np.ones(t.shape),t,np.sin(2*np.pi*np.arange(len(t))*1000),np.cos(2*np.pi*np.arange(len(t))*1000)])
#Xtrain,Xtest,Ytrain,Ytest=train_test_split(X.T,t)
#clf = LinearRegression();clf.fit(Xtrain,Ytrain)
#plt.figure();plt.plot(Ytest);plt.plot(clf.predict(Xtest))