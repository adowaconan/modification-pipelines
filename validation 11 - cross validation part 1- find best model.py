# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:12:36 2017

@author: install
"""

import numpy as np
import pickle
from random import shuffle
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import os


os.chdir('D:\\NING - spindle\\training set')
threshold = 0.15
types = 'with'
samples = pickle.load(open("temp_data\\%.2f_samples_%s.p"%(threshold,types),
                                "rb"))
label = pickle.load(open("temp_data\\%.2f_labels_%s.p" %(threshold,types),
                        "rb"))
data,label,idx_row = np.array(samples),np.array(label),np.arange(0,len(label),1)
for ii in range(100):
    shuffle(idx_row)
data,label = data[idx_row,:],label[idx_row]
X_train, X_test, y_train, y_test = train_test_split(data,label,train_size=0.75)
tpot = TPOTClassifier(generations=5, population_size=20,
                      verbosity=2,random_state=0,num_cv_folds=3 )
tpot.fit(X_train,y_train)
tpot.score(X_test,y_test)
tpot.export('tpot_exported_pipelien.py')    


        
        
    
    
