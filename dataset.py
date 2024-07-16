from torch.utils.data import Dataset
import numpy as np
import torch

class FittingDataset(Dataset):  
    def __init__(self):  
        X_train = np.load("data/X_train.npy")
        y_train = np.load("data/y_train.npy")
        self.X_train = torch.tensor(X_train, dtype=torch.float32)  
        self.y_train = torch.tensor(y_train, dtype=torch.float32) 
  
    def __len__(self):  
        return len(self.y_train)  
  
    def __getitem__(self, idx):  
        return self.X_train[idx], self.y_train[idx]