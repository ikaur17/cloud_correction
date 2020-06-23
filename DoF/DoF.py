#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:17:11 2020

@author: inderpreet
"""

import numpy as np
import os
from TB_aws import TB_AWS
from tabulate import tabulate


channels_3 =  ['C21','C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C41', 'C42', 'C43']
channels_4 =  ['C21','C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C41', 'C42', 'C43', 'C44']
channels_4X = ['C21','C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C4X']

file_3a  = '~/Dendrite/Projects/AWS-325GHz/TB_AWS/TB_AWS_m60_p60.nc'
file_3b  = '~/Dendrite/Projects/AWS-325GHz/TB_AWS/TB_AWS_m60_p60_three_b.nc'
file_4  = '~/Dendrite/Projects/AWS-325GHz/TB_AWS/TB_AWS_m60_p60_four.nc'
file_4X = '~/Dendrite/Projects/AWS-325GHz/TB_AWS/TB_AWS_m60_p60.nc'

files    = [file_3a, file_3b,  file_4, file_4X]
T_recs   = [None, 1, 2]
channels = [channels_3, channels_3, channels_4, channels_4X]

options = ["3a", "3b", "4", "3b"]


all_cases = False
cloudy    = True
clear     = False

DoF = np.zeros([4, 3])



for i, (channel, file, option) in enumerate(zip(channels, files, options)):
#    print (channel, file, option)
    for j , T_rec in enumerate(T_recs):
        
 #       print (i, j,  T_rec)
        Y = TB_AWS(os.path.expanduser(file),
                    inChannels = channel, 
                    option     = option, 
                    T_rec      = T_rec, 
                    all_cases  = all_cases,
                    cloudy     = cloudy,
                    clear      = clear)

#        print (Y.index_183)

        s_epsilon = Y.add_noise(Y)
    
        U, S, V = Y.svd()
        S_l = Y.get_S_lambda(U, s_epsilon)
        
        S_y = Y.cov_mat()
        
        DoF[i,j]= len(np.where(S > S_l.diagonal())[0])
        print (channel, option, T_rec)
        print (np.where(S > S_l.diagonal())[0])
#        print (T_rec, channel, DoF[i,j])
    

sets = ["3a", "3b", "4"]
table  = [[sets[i], DoF[i, 0], DoF[i, 1], DoF[i, 2]] for i in range(3)]

print(tabulate(table
         , ['T_rec = 1200 K', 'T_rev = 1800 K', 'T_rec = 2400 K'],  tablefmt="latex"))


    



