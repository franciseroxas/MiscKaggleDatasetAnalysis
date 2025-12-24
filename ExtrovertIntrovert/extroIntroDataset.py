import torch
from torch.utils.data import Dataset
from loadAndCleanUpDataset import loadAndCleanUpDataset, turnDatasetIntoTorchTensor

class extroIntroDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        return

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        #Get the entire row related to the data
        return self.x[idx, :], self.y[idx]
