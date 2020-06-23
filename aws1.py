import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset

class awsData(Dataset):
    """
    Pytorch dataset for the AWS training data.

    """
    def __init__(self, path, inChannels, batch_size = None):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size

        """
        super().__init__()
        self.batch_size = batch_size

        self.file = netCDF4.Dataset(path, mode = "r")

#        TB = self.file.variables["TB_noise"][:]
        TB = self.file.variables["TB"][:]
        channels = self.file.variables["channels"][:]

 #       print (channels)
        self.channels = inChannels
        i1, = np.argwhere(channels == inChannels[0])[0] 
        i2, = np.argwhere(channels == inChannels[1])[0]     
        i3, = np.argwhere(channels == inChannels[2])[0]     
        i4, = np.argwhere(channels == inChannels[3])[0]
        
        self.index = [i1, i2, i3, i4]
        
        C1 = TB[i1, 1, :]        
        C2 = TB[i2, 1, :]
        C3 = TB[i3, 1, :]
        C4 = TB[i4, 1, :]

        #store mean and std to normalise data  
        
        self.x = np.float32(np.stack([C1, C2, C3, C4], axis = 1))
        
        x_noise = self.add_noise(self.x)
  
        self.std = np.std(x_noise, axis = 0)
        self.mean = np.mean(x_noise, axis = 0)   

        
        self.y = np.float32(TB[i1, 0, :])
        
        self.x = self.x.data
        self.y = self.y.data
        

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.x.shape[0]
        else:
            return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i: The index of the sample to return
        """
        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]),
                    torch.tensor(self.y[[i]]))
        else:
            i_start = self.batch_size * i
            i_end = self.batch_size * (i + 1)    
            
            x  = self.x[i_start : i_end, :].copy()
            
            
            x_noise = np.float32(self.add_noise(x))
            x_norm = np.float32(self.normalise(x_noise))
            
            
            return (torch.tensor(x_norm),
                    torch.tensor(self.y[i_start : i_end]))
        
        
    def add_noise(self, x):        
        """
        Gaussian noise is added to every measurement before before used 
        for training again
        
        Args: 
            the input TB in one batch of size (batch_size x number of channels)
        Returns:
            input TB with noise
            
        """
        c = 1.2
        delta_t = 0.003
        T_rec = np.array([390., 650., 650.,650., 650., 650., 650., 1000., 
                          1200., 1200., 1200.])
        
        bandwidth_89 = np.array([4000]) * 1e6
        bandwidth_165 = np.array([2800]) * 1e6
        bandwidth_183 = np.array([2000, 2000, 1000, 1000, 500])*1e6
        bandwidth_229 = np.array([2000])*1e6
        bandwidth_325 = np.array([3000, 2500, 1200])* 1e6
        delta_f = np.concatenate([bandwidth_89, bandwidth_165, bandwidth_183,
                                  bandwidth_229, bandwidth_325])

        # for the channels we need        
        T_rec_subset = T_rec[self.index]

        delta_f_subset = delta_f[self.index]
    
        cases = x.shape[0]
        for ic in range(len(self.channels)):
            T_a = x[:, ic]
            sigma = c * (T_rec_subset[ic] + T_a)/np.sqrt(delta_f_subset[ic] * delta_t)

            noise = np.random.normal(0, 1, cases) * sigma

            x[:, ic] = T_a + noise
        return x           
 
    def normalise(self, x):
        
        x_norm = (x - self.mean)/self.std
            
        return x_norm 


