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
# def get_logger(output_file):
#     log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
#     logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
#     while os.path.exists(output_file):
#         output_file = output_file.replace('.log', '1.log')
#     if output_file:
#         logger.add(output_file, enqueue=True, format=log_format)
#     return logger
#
# def parse_arguments():
#     parser = argparse.ArgumentParser("Training Parameters.")
#     parser.add_argument('--n_hidden', type=int, default=1024, help='the num of hidden neurons')
#     parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
#     parser.add_argument('--batch', type=int, default=128, help='batch_size')
#     args = parser.parse_args()
#     return args
#
# args = parse_arguments()
# output = f'task1/train_{datetime.now():%Y-%m-%d_%H:%M:%S}.log'
# loger = get_logger(output)
# loger.info(args)

test_path = './schneider50k/raw_test.csv'
test_data = pd.read_csv(test_path)
test_reactions = test_data['reactants>reagents>production']
train_path = './schneider50k/raw_train.csv'
train_data = pd.read_csv(train_path)
train_reactions = train_data['reactants>reagents>production']

test_Templates = []  # reactants
test_FingerPrints = [] #morgan fingerprints
test_Products = [] # products

for reaction in test_reactions:
    reactants, products = reaction.split('>>')
    testRec = {'_id': None, 'reactants': reactants, 'products': products}
    ans = extract_from_reaction(testRec)
    if 'reaction_smarts' in ans.keys():
        test_Templates.append(ans['reaction_smarts'])
        test_Products.append(products)
        # print(ans['reaction_smarts'])
    mol = Chem.MolFromSmiles(products)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1
    test_FingerPrints.append(arr)

train_Templates = []
train_FingerPrints = []
train_Products = []
for reaction in train_reactions:
    reactants, products = reaction.split('>>')
    trainRec = {'_id': None, 'reactants': reactants, 'products': products}
    ans = extract_from_reaction(trainRec)
    if 'reaction_smarts' in ans.keys():
        train_Templates.append(ans['reaction_smarts'])
        train_Products.append(products)
        # print(ans['reaction_smarts'])
    mol = Chem.MolFromSmiles(products)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1
    train_FingerPrints.append(arr)

length1 = max(len(string) for string in train_Products) if train_Products else 0
length2 = max(len(string) for string in test_Products) if test_Products else 0

max_len = max(length1, length2)

template_dict = {}
i = 0
for template in train_Templates:
    if template not in template_dict.keys():
        template_dict[template] = i
        i+=1
for template in test_Templates:
    if template not in template_dict.keys():
        template_dict[template] = i
        i+=1
nums = len(template_dict)
train_data = []
for product in train_Products:
    data = []
    for char in product:
        data.append(ord(char))
    if len(data) < max_len:
        for k in range(max_len-len(data)):
            data.append(0)
    train_data.append(data)
train_data = np.array(train_data, dtype=float)
train_label = []
for template in train_Templates:
    train_label.append(template_dict[template])
train_label = np.array(train_label)

np.save('train_data.npy', train_data)
np.save('train_label.npy', train_label)
test_data = []
for product in test_Products:
    data = []
    for char in product:
        data.append(ord(char))
    if len(data) < max_len:
        for k in range(max_len-len(data)):
            data.append(0)
    test_data.append(data)
test_data = np.array(test_data, dtype=float)

test_label = []
for template in test_Templates:
    test_label.append(template_dict[template])
test_label = np.array(test_label)
np.save('test_data.npy', test_data)
np.save('test_label.npy', test_label)

print(max_len)
print(nums)
print(i)
# mlp = nn.Sequential(
#     nn.Linear(max_len, args.n_hidden),
#     nn.Tanh(),
#     nn.Linear(args.n_hidden, 2*args.n_hidden),
#     nn.Tanh(),
#     nn.Linear(2*args.n_hidden, nums)
# )
#
# device = torch.device('cuda:0')
# random.seed(114514)
# np.random.seed(114514)
# torch.manual_seed(114514)
# model = mlp.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# loss_func = nn.CrossEntropyLoss()
# train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=args.batch)
#
# final_acc = 0.0
# for epoch in range(100):
#     losses = []
#     acc = 0.0
#     num = 0
#     for batch in train_loader:
#         result = model(batch[0].to(device))
#         loss = loss_func(result, batch[1].long().to(device))
#         losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#     loger.info(f"epoch={epoch}, train_loss, {sum(losses) / len(losses)}")
#
#     with torch.no_grad():
#         correct = 0
#         all = 0
#         for batch in test_loader:
#             result = model(batch[0].to(device))
#             pred = result.argmax(dim=-1)
#             correct += (pred == batch[1].to(device)).sum().item()
#             all += len(pred)
#             acc += correct / all
#             num += 1
#         loger.info(f'test, acc={correct/all}')
#     acc = acc/num
#     loger.info(f'epoch:{epoch},test_acc={acc}')
#     final_acc = max(acc, final_acc)
#     loger.info(f'final_acc={final_acc}')
#
