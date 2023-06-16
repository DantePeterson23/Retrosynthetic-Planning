import os.path
import argparse
import sys
import numpy as np
from datetime import datetime
from loguru import logger
from glob import glob
import pdb
import  torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.labels = label

    def __getitem__(self, index):
        return torch.tensor(self.data[index], dtype=torch.float32), torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.labels)