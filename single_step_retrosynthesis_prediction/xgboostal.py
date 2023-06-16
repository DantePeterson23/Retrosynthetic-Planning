import pdb
from loguru import logger
import time
import os
import os.path as osp
import pandas as pd
import numpy as np
# from rdchiral.template_extractor import extract_from_reaction
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from dataset import MyDataset
# import torch.nn as nn
# import torch
# from torch.utils.data import DataLoader, TensorDataset, Dataset
# from torch import autograd
# import torch.nn.functional as F
# from torch.distributions import Categorical
from sklearn.decomposition import PCA
import sys
from datetime import datetime
import argparse
import xgboost
from xgboost import XGBClassifier
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger

output = 'result.log'
loger = get_logger(output)
scaler = StandardScaler()
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')
train_data_fin = np.load('train_data_fin.npy')
test_data_fin = np.load('test_data_fin.npy')
test_label_fin = np.load('test_label_fin.npy')
train_label_fin = np.load('train_label_fin.npy')
pca1 = PCA(n_components=50)
pca2 = PCA(n_components=200)
xgb1 = XGBClassifier()
xgb2 = XGBClassifier()

pca1.fit(train_data)
train_data = pca1.transform(train_data)
test_data = pca1.transform(test_data)
xgb1.fit(train_data, train_label)
pred1 = xgb1.predict(test_data)
acc1 = accuracy_score(test_label, pred1)
loger.info(acc1)


pca2.fit(train_data_fin)
train_data_fin = pca2.transform(train_data_fin)
test_data_fin = pca2.transform(test_data_fin)
xgb2.fit(train_data_fin, train_label_fin)
pred2 = xgb2.predict(test_data_fin)
acc2 = accuracy_score(test_label_fin, pred2)
loger.info(acc2)