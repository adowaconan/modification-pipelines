# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:51:04 2017

@author: ning
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
plt.rcParamsDefault.update({'axes.labelsize':320,'xtick.labelsize':32,'ytick.labelsize':32})
data = pd.read_csv('drive data.csv')
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
plt.setp(ax.yaxis.get_majorticklabels(),rotation=60)

sns.regplot(xvar[0],yvar[0],data=data,robust=True)


