import pdb
from loguru import logger
import time
import os
import os.path as osp
import pandas as pd
import numpy as np
from rdchiral.template_extractor import extract_from_reaction
from rdkit import Chem
from rdkit.Chem import AllChem
from dataset import MyDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import autograd
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
from datetime import datetime
import argparse
import random
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
def get_logger(output_file):
    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    while os.path.exists(output_file):
        output_file = output_file.replace('.log', '1.log')
    if output_file:
        logger.add(output_file, enqueue=True, format=log_format)
    return logger

def parse_arguments():
    parser = argparse.ArgumentParser("Training Parameters.")
    parser.add_argument('--n_hidden', type=int, default=4096, help='the num of hidden neurons')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch', type=int, default=128, help='batch_size')
    args = parser.parse_args()
    return args

args = parse_arguments()
output = f'task1/train_{datetime.now():%Y-%m-%d_%H:%M:%S}.log'
loger = get_logger(output)
loger.info(args)
scaler = StandardScaler()
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')
test_data = np.load('test_data.npy')
test_label = np.load('test_label.npy')
train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)
train_set = MyDataset(train_data, train_label)
test_set = MyDataset(test_data, test_label)
# pdb.set_trace()
# pdb.set_trace()
mlp = nn.Sequential(
    nn.Linear(708, args.n_hidden),
    nn.ReLU(),
    nn.Linear(args.n_hidden, 12578),
)

device = torch.device('cuda:0')
random.seed(114514)
np.random.seed(114514)
torch.manual_seed(114514)
model = mlp.to(device)

def train_test(learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch)

    final_acc = 0.0
    for epoch in range(50):
        losses = []
        acc = 0.0
        num = 0
        for batch in train_loader:
            result = model(batch[0].to(device))
            loss = loss_func(result, batch[1].long().to(device))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        loger.info(f"epoch={epoch}, train_loss, {sum(losses) / len(losses)}")

        with torch.no_grad():
            correct = 0
            all = 0
            for batch in test_loader:
                result = model(batch[0].to(device))
                pred = result.argmax(dim=-1)
                correct += (pred == batch[1].to(device)).sum().item()
                all += len(pred)
                acc += correct / all
                num += 1
            loger.info(f'test, acc={correct / all}')
        acc = acc / num
        loger.info(f'epoch:{epoch},test_acc={acc}')
        final_acc = max(acc, final_acc)
        loger.info(f'final_acc={final_acc}')
    return final_acc

accs = []
lrs = []
for lr in [0.0001,0.0002, 0.0005, 0.0008, 0.001, 0.005, 0.01]:
    acc = train_test(lr)
    lrs.append(lr)
    accs.append(acc*100)
plt.xlabel('learning_rate')  # x轴标题
plt.ylabel('accuracy(%)')  # y轴标题
plt.plot(lrs, accs, marker='o', markersize=3)
plt.savefig('acc.jpg')