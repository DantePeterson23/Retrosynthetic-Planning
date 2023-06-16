import random
import dill
import numpy as np
import argparse
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=100000)
parser.add_argument('--test_size', type=int, default=1000)
parser.add_argument('--method', type=str, default='neural_network',
                    help='Choose among regression, cosine_similarity and neural_network')
parser.add_argument('--type', type=str, default='single', help='Choose between single and multiple')
parser.add_argument('--equation', type=int, default=1, help='Choose between 1 and 2')


class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(512, 32)
        self.activation2 = nn.ReLU()
        self.last_layer = nn.Linear(32, 1)
        self.last_activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.last_layer(x)
        x = self.last_activation(x)
        return x.squeeze(-1)


class MyDataset(Dataset):
    def __init__(self, data, label):
        super(MyDataset, self).__init__()
        self.data = torch.tensor(np.array(data), dtype=torch.float32)
        self.data = self.data.reshape(-1, 2048)
        self.label = torch.tensor(np.array(label), dtype=torch.float32)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)


class Model():
    def __init__(self, input_dim, train_loader, test_loader):
        self.mlp = MLP(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=1e-3)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_and_test(self, type='single'):
        for epoch in range(10):
            Loss = 0
            for i, (data, label) in enumerate(self.train_loader):
                self.mlp.train(True)
                self.optimizer.zero_grad()
                y = self.mlp(data)
                loss = self.criterion(y, label)
                Loss += loss
                loss.backward()
                self.optimizer.step()
            print('Epoch = %d, Training loss = %f' % (epoch, Loss / len(train_dataset)))
            Loss = 0
            Accurate = 0
            for i, (data, label) in enumerate(self.test_loader):
                self.mlp.eval()
                y = self.mlp(data).detach()
                loss = self.criterion(y, label)
                Loss += loss
                for j in range(len(label)):
                    if pow(y[j] - label[j], 2) <= 1e-2:
                        Accurate += 1
            print('Epoch = %d, Testing loss = %f, Accuracy = %f' % (
                epoch, Loss / len(test_dataset), Accurate / len(test_dataset)))

    def multi_molecule_evaluation(self, data, label, equation):
        if equation == 1:
            cost = self.mlp(data).detach().sum()
            loss = pow(cost - label, 2)
            print('Predicted cost: %f, actual cost: %f, loss: %f' % (cost.item(), label, loss.item()))
            return loss.item()
        if equation == 2:
            cost = self.mlp(data).detach()
            Loss = 0
            for i in range(len(cost)):
                loss = pow(cost[i] - label[i], 2)
                print('Predicted cost: %f, actual cost: %f, loss: %f' % (cost[i], label[i], loss))
                Loss += loss
            return Loss


if __name__ == '__main__':
    # 1. 数据提取，获得训练集与测试集
    with open('./MoleculeEvaluationData/train.pkl', 'rb') as f:
        train_data = dill.load(f)

    packed_fps = train_data['packed_fp']
    Values = train_data['values']
    train_FingerPrints = []
    for i in range(len(packed_fps)):
        unpacked_fp_i = np.unpackbits(packed_fps[i])
        train_FingerPrints.append(unpacked_fp_i)

    train_Costs = []
    for i in range(len(Values)):
        train_Costs.append(float(Values[i][0]))

    with open('./MoleculeEvaluationData/test.pkl', 'rb') as f:
        test_data = dill.load(f)

    packed_fps = test_data['packed_fp']
    Values = test_data['values']
    test_FingerPrints = []
    for i in range(len(packed_fps)):
        unpacked_fp_i = np.unpackbits(packed_fps[i])
        test_FingerPrints.append(unpacked_fp_i)

    test_Costs = []
    for i in range(len(Values)):
        test_Costs.append(float(Values[i][0]))

    args = parser.parse_args()
    # 设置使用的训练集与测试集大小
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    if args.type == 'single':  # single molecule evaluation
        # 使用余弦相似度预测分子 cost
        if args.method == 'cosine_similarity':
            Loss = 0
            Accurate = 0
            for i in range(0, TEST_SIZE):
                similarity = cosine_similarity(test_FingerPrints[i].reshape(1, -1), train_FingerPrints[0: TRAIN_SIZE])[0]
                best = int(np.argmax(similarity))
                loss = pow(train_Costs[best] - test_Costs[i], 2)
                print('Test sample: %d, Best fit sample: %d, Predicted cost: %f, Actual cost: %f, Square loss: %f' %
                      (i, best, train_Costs[best], test_Costs[i], loss))
                Loss += loss
                if loss <= 1e-2:
                    Accurate += 1
            print('Average loss:', Loss / TEST_SIZE, ', Accuracy:', Accurate / TEST_SIZE)

        # 使用线性回归预测分子 cost
        if args.method == 'regression':
            Accurate = 0
            regression = LinearRegression()
            regression.fit(train_FingerPrints[0:TRAIN_SIZE], train_Costs[0:TRAIN_SIZE])
            score = regression.score(test_FingerPrints[0:TEST_SIZE], test_Costs[0:TEST_SIZE])
            predict_train = regression.predict(train_FingerPrints[0:TRAIN_SIZE])
            predict_test = regression.predict(test_FingerPrints[0:TEST_SIZE])
            loss_train = np.sum(np.square(predict_train - train_Costs[0:TRAIN_SIZE])) / TRAIN_SIZE
            loss_test = np.sum(np.square(predict_test - test_Costs[0:TEST_SIZE])) / TEST_SIZE
            for i in range(TEST_SIZE):
                loss = pow(predict_test[i] - test_Costs[i], 2)
                print('Test sample: %d, Predicted cost: %f, Actual cost: %f, Square loss: %f' %
                      (i, predict_test[i], test_Costs[i], loss))
                if loss <= 1e-2:
                    Accurate += 1
            accuracy = Accurate / TEST_SIZE

            print('loss_train:', loss_train, 'loss_test:', loss_test, 'accuracy:', accuracy, 'score:', score)

        # 使用深度回归模型预测分子 cost
        if args.method == 'neural_network':
            train_dataset = MyDataset(train_FingerPrints[0:TRAIN_SIZE], train_Costs[0:TRAIN_SIZE])
            test_dataset = MyDataset(test_FingerPrints[0:TEST_SIZE], test_Costs[0:TEST_SIZE])
            train_loader = DataLoader(train_dataset, batch_size=16)
            test_loader = DataLoader(test_dataset, batch_size=16)
            model = Model(2048, train_loader, test_loader)
            model.train_and_test()

    elif args.type == 'multiple':
        # 采样
        SAMPLE_NUM = 10000
        TEST_RATIO = 0.2
        Data = []
        Label = []
        for i in range(SAMPLE_NUM):
            molecule_num = random.randint(2, 5)  # 分子数量
            molecule_ids = random.sample(range(TRAIN_SIZE), molecule_num)  # 分子采样编号
            finger = []  # 将各指纹拼接起来
            cost = 0
            for id in molecule_ids:
                finger.append(train_FingerPrints[id])
                cost += train_Costs[id]
            Data.append(finger)
            Label.append(cost)

        if args.equation == 1:  # 1. 直接将各分子cost相加
            # 用单分子训练 mlp
            train_dataset = MyDataset(train_FingerPrints[0:TRAIN_SIZE], train_Costs[0:TRAIN_SIZE])
            test_dataset = MyDataset(test_FingerPrints[0:TEST_SIZE], test_Costs[0:TEST_SIZE])
            train_loader = DataLoader(train_dataset, batch_size=16)
            test_loader = DataLoader(test_dataset, batch_size=16)
            model = Model(2048, train_loader, test_loader)
            model.train_and_test()
            Loss = 0
            for i in range(int(TEST_RATIO * SAMPLE_NUM)):
                loss = model.multi_molecule_evaluation(torch.tensor(np.array(Data[i]).reshape(-1, 2048), dtype=torch.float32),
                                                        torch.tensor(np.array(Label[i]), dtype=torch.float32), 1)
                Loss += loss
            print(Loss / int(TEST_RATIO * SAMPLE_NUM))
        if args.equation == 2:  # 2. 将各分子合在一起计算（此处采用指纹求和的形式）
            train_data = []
            for data in Data:
                train_data.append(list(map(sum, zip(*data))))

            split = int(SAMPLE_NUM * (1 - TEST_RATIO))
            train_dataset = MyDataset(train_data[0:split], Label[0:split])
            test_dataset = MyDataset(train_data[(split):-1], Label[(split):-1])
            train_loader = DataLoader(train_dataset, batch_size=16)
            test_loader = DataLoader(test_dataset, batch_size=16)
            model = Model(2048, train_loader, test_loader)
            model.train_and_test()
            Loss = model.multi_molecule_evaluation(torch.tensor(np.array(train_data[(split):-1]), dtype=torch.float32),
                                                    torch.tensor(np.array(Label[(split):-1]), dtype=torch.float32), 2)
            print(Loss / int(TEST_RATIO * SAMPLE_NUM))