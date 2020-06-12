import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset

class awsTestData():
    """
    Pytorch dataset for the AWS training data.

    """
    def __init__(self, path, inChannels, option):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size

        """


        path0 = path.replace("_noise", "")

        self.file = netCDF4.Dataset(path, mode = "r")
        self.file0 = netCDF4.Dataset(path0, mode = "r")
        
        TB = self.file.variables["TB_noise"][:]
        TB0 = self.file0.variables["TB"][:]
        channels = self.file.variables["channels"][:]

 #       print (channels)
        self.channels = inChannels
        self.option = option
#       find index for input channels
        

        i1, = np.argwhere(channels == inChannels[0])[0] 
        i2, = np.argwhere(channels == inChannels[1])[0]     
        i3, = np.argwhere(channels == inChannels[2])[0]     
        i4, = np.argwhere(channels == inChannels[3])[0]
        
        if self.option == 4:
            i5, = np.argwhere(channels == inChannels[4])[0]        
            self.index = [i1, i2, i3, i4, i5]
        else:
            self.index = [i1, i2, i3, i4]
        

        C1 = TB[i1, 1, :]        
        C2 = TB[i2, 1, :]
        C3 = TB[i3, 1, :]
        C4 = TB[i4, 1, :]
        if self.option == 4:
            C5 = TB[i5, 1, :]

        x = np.float32(np.stack([C1, C2, C3, C4], axis = 1))
        if self.option == 4:
            x = np.float32(np.stack([C1, C2, C3, C4, C5], axis = 1))
        
        #store mean and std to normalise data  
#        x_noise = self.add_noise(self.x)
  
        self.std = np.std(x, axis = 0)
        self.mean = np.mean(x, axis = 0)   
        
#       noise free clear sky values        
        self.y0 = np.float32(TB0[i1, 0, :])
#       noisy clear sky values
        self.y = np.float32(TB[i1, 0, :])
 #       self.x = self.normalise(x)
        self.x  = x.data
        self.y  = self.y.data
        self.y0 = self.y0.data
        
        
 
    def normalise(self, x):
        """
        normalise the input data wit mean and standard deviation
        Args:
            x
        Returns :
            x_norm
        """            
        x_norm = (x - self.mean)/self.std   
            
        return x_norm 


