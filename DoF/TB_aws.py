#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:33:30 2020

@author: inderpreet
"""
import numpy as np
import netCDF4
 
class TB_AWS():
    """
    CLass for the AWS test data.

    """
    def __init__(self, path, inChannels, option, T_rec, all_cases = False, 
                 cloudy = False, clear = False  ):
        """
        Read AWS data for channels in "inChannels" 

        Args:
            path: Path to the NetCDF4 containing the data.
            inChannels : The list of channels to be used

        """
        self.option = option
        self.T_rec = T_rec
        self.file = netCDF4.Dataset(path, mode = "r")

        TB = self.file.variables["TB"][:]
        channels = self.file.variables["channels"][:]
        channels_183 = ["C32", "C33", "C34", "C35", "C36"]
        self.channels = inChannels
        self.index = []
        
        for c in self.channels:
            self.index.append(np.argwhere(channels == c)[0,0])
            
        self.index_183 = []
        
        for c in channels_183:
            self.index_183.append(np.argwhere(channels == c)[0,0])    

        C = []

        
        for i in range(len(self.channels)):
            C.append(TB[self.index[i], :, :])
            
        Y = np.float32(np.stack(C, axis = 1))
        Y = Y[:, :].data
        
        if all_cases:
            # use all cases
            self.Y = Y[1, :, :]
            self.Y = np.transpose(self.Y)
            self.mean = np.mean(self.Y, axis = 0)
            self.n = self.Y.shape[0]  
        
        if clear:
            # subsets the vector Y, if only clear values are to be used
            # a case is cloudy if cloud impact is greater than 1 K in any of 183GHz channels
            
            i_start = self.index_183[0]
            i_end   = self.index_183[-1]
            cloud_imp = np.abs(Y[1, i_start:i_end, :] - Y[0, i_start:i_end, :])
            icloud = cloud_imp < 1.0
            icloud = np.any(icloud, axis = 0)
        
            self.Y = Y[1, :, icloud]
            self.mean = np.mean(self.Y, axis = 0)
            self.n = self.Y.shape[0]
            
        if cloudy:
            #subsets the vector Y, if only cloudy values are to be used
            
            i_start = self.index_183[0]
            i_end   = self.index_183[-1]
            cloud_imp = np.abs(Y[1, i_start:i_end+1, :] - Y[0, i_start:i_end+1, :])

            icloud = cloud_imp >= 1.0
            print(np.sum(icloud[:, :], axis = 1))
            icloud = np.any(icloud, axis = 0)
        
            self.Y = Y[1, :, icloud]
            self.mean = np.mean(self.Y, axis = 0)
            self.n = self.Y.shape[0]
            
        
        
    def add_noise(self, x):        
        """
        Gaussian noise is added for TB = 250K for each channel
        
        Args: 
        
        Returns:
            
            
        """
        c = 1.2
        delta_t = 0.003

        T_rec = np.array([390., 650., 650.,650., 650., 650., 650., 1000., 
                          1200., 1200., 1200., 1200.])
        
        if self.T_rec == 1:
            T_rec = np.array([390., 650., 650.,650., 650., 650., 650., 1000., 
                          1800., 1800., 1800., 1800.])
            
        if self.T_rec == 2:
            T_rec = np.array([390., 650., 650.,650., 650., 650., 650., 1000., 
                          2400., 2400., 2400., 2400.])           
        
        bandwidth_89 = np.array([4000]) * 1e6
        bandwidth_165 = np.array([2800]) * 1e6
        bandwidth_183 = np.array([2000, 2000, 1000, 1000, 500])*1e6
        bandwidth_229 = np.array([2000])*1e6
        
        if self.option == "3a":
            bandwidth_325 = np.array([3000, 2500, 1200])* 1e6
            
        if self.option == "3b":
            bandwidth_325 = np.array([3000, 3000, 800])* 1e6    
            
        if self.option == "4":
            bandwidth_325 = np.array([2800, 1800, 1200, 800])*1e6
        delta_f = np.concatenate([bandwidth_89, bandwidth_165, bandwidth_183,
                                  bandwidth_229, bandwidth_325])

        # for the channels we need        
        T_rec_subset = T_rec[self.index]

        delta_f_subset = delta_f[self.index]
    
        noise = []
        for ic in range(len(self.channels)):
            T_a = 250.0
            sigma = c * (T_rec_subset[ic] + T_a)/np.sqrt(delta_f_subset[ic] * delta_t)

            noise.append(sigma ** 2)        
            
        n = len(noise)
        s_epsilon = np.zeros([n , n])
        for i in range(n):
            s_epsilon[i, i] = noise[i]
        return s_epsilon   

      
        

    def cov_mat(self):
        """
        S_y = \frac{1}{n-1} * YY^T
        where Y is the ensemble of n measurements with mean subtracted 

        Returns
        -------
        S_y, covariance matrix 

        """
        
        Y = self.Y - self.mean
        Y = np.transpose(Y)
        S_y = (np.dot(Y, np.transpose(Y)))/(self.n - 1)
        
        return S_y
 

    def svd(self):
        """
        Singular value decomposition of Y, to estimate the eigenvectors
        USVh = Y

        Returns
        U, S, Vh
        -------
        None.

        """
        Y = self.Y - self.mean
#        print (Y)
        Y = np.transpose(Y)/np.sqrt(self.n - 1)
        
        U, S, V = np.linalg.svd(Y, full_matrices = False)
        
        return U, S, V


    def get_S_lambda(self, U, noise):
        """
        
        S_\Lambda = ES_\epsilonE^T
        where, E is the eigenvector of Y, and S_\epsilon is NEDT in diagonal elements
        Returns
        -------
        None.

        """           
        S_lambda = np.dot(U, np.dot(noise, np.transpose(U)))
        return S_lambda                 
    