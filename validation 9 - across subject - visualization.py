# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 16:39:59 2017

@author: install
"""

import pickle
import matplotlib.pyplot as plt
import eegPinelineDesign
import numpy as np
file_in_fold=eegPinelineDesign.change_file_directory('D:\\NING - spindle\\training set')
with open('over_step_size.p', 'rb') as handle:
    data = pickle.load(handle)
with_sleep_stage=np.array(data['with'])
without_sleep_stage=np.array(data['without'])

a = with_sleep_stage.mean(1)[with_sleep_stage.mean(1)[:,0].argsort()]
plt.plot(a[:,0],a[:,1],'b-')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',alpha=0.1) 