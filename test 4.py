# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:58:38 2017

@author: ning
"""

working_dir = 'D:\\Epochs\\'
import os
os.chdir(working_dir)
import mne
import numpy as np
epochs = []
epoch = [mne.read_epochs('Ex10_Suj19_Run%d-epo.fif'% ii) for ii in np.arange(1,5)]
epochs = mne.concatenate_epochs(epoch)
old = epochs['after'].average()
new_new = epochs['new'].average()
new_old = epochs['before'].average()
scramble = epochs['scramble'].average()
mne.combine_evoked([old, -new_old],weights='equal').plot_joint(times=[0,.4,.8,1.2],title='old vs [11,21,31]')
mne.combine_evoked([old, -new_new],weights='equal').plot_joint(times=[0,.4,.8,1.2],title='old vs new' )
mne.combine_evoked([old, -scramble],weights='equal').plot_joint(times=[0,.4,.8,1.2],title='old vs scramble')
old.pick_channels(['PO8']).plot(titles='old')
new_new.pick_channels(['PO8']).plot(titles='new')