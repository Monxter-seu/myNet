#!/usr/bin/env python
# Created on 2023/03
# Author: HUA

import torch
import torch.nn as nn
import os
import numpy as np
import g_mlp
import librosa
import data
import pit_criterion
import torch.optim as optim
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from g_mlp import gMLP
from pit_criterion import new_loss
from data import MyDataLoader, MyDataset




group_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# print(tensor_read)
# print(tensor_read.shape)

if __name__ == "__main__":


    # with open(address+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
    # reader = csv.reader(file)
    # row = next(reader)
    # tensor_read = torch.from_numpy(np.array(row, dtype=np.float32))
    N = 128
    L = 20
    B = 128
    H = 32
    P = 3
    X = 8
    R = 2
    norm_type = 'gLN'
    causal = 0
    mask_nonlinear = 'relu'
    C = 2

    # 实例化模型

    model = gMLP(N, L, B, H, P, X, R, C, norm_type=norm_type)
    model = model.cuda()

    # 定义损失函数和优化器
    # criterion = new_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_dataset = MyDataset('D:\\csvProcess\\testout\\tr\\', batch_size=16)
    train_loader = MyDataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    cv_dataset = MyDataset('D:\\csvProcess\\testout\\tt\\', batch_size=16)
    cv_loader = MyDataLoader(cv_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataset = MyDataset('D:\\csvProcess\\testout\\cv\\', batch_size=16)
    test_loader = MyDataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)
    # 训练模型
    num_epochs = 30
    for epoch in range(num_epochs):
        # 训练模式
        #start_time = time.time()
        #count = 0
        model.train()
        train_loss = 0
        train_correct0 = 0
        train_correct1 = 0
        total_samples = 0
        # print('train_loader.shape',train_loader.shape)
        for data, labels in tqdm(train_loader):
            # 将数据和标签转换为张量
            data = data.cuda()
            labels =labels.cuda()

            # 向前传递
            outputs = model(data)
            loss = new_loss(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            splited_outputs0 = outputs[:, 0].unsqueeze(1)
            splited_outputs1 = outputs[:, 1:7]

            # 计算损失和准确率
            train_loss += loss.item() * data.size(0)
            predicted0 = (splited_outputs0 > 0.5).float()
            _, predicted1 = torch.max(splited_outputs1, 1)
            train_correct0 += (predicted0 == labels[:,0].unsqueeze(1)).sum().item()
            train_correct1 += (predicted1 == labels[:,1]).int().sum().item()
            total_samples += labels.size(0)
            
            #count++

            
        train_loss /= len(train_loader.dataset)
        train_accuracy0 = train_correct0 / total_samples
        train_accuracy1 = train_correct1 / total_samples
        print('============')
        print('train_loss', train_loss)
        print('train_accuracy0', train_accuracy0)
        print('train_accuracy1', train_accuracy1,flush=True)  
        
        
        model.eval()
        cv_loss = 0
        cv_correct0 = 0
        cv_correct1 = 0
        total_samples = 0
        with torch.no_grad():
            for data, labels in cv_loader:
                # 将数据和标签转换为张量
                data = data.cuda()
                labels = labels.cuda()
        
                # 向前传递
                outputs = model(data)
                loss = new_loss(outputs, labels)

                splited_outputs0 = outputs[:, 0].unsqueeze(1)
                splited_outputs1 = outputs[:, 1:7]
        
                cv_loss += loss.item() * data.size(0)
                predicted0 = (splited_outputs0 > 0.5).float()
                _, predicted1 = torch.max(splited_outputs1, 1)
                cv_correct0 += (predicted0 == labels[:,0].unsqueeze(1)).sum().item()
                cv_correct1 += (predicted1 == labels[:,1]).int().sum().item()
                total_samples += labels.size(0)
            cv_loss /= len(cv_loader.dataset)
            cv_accuracy0 = cv_correct0 / total_samples
            cv_accuracy1 = cv_correct1 / total_samples

            print('cv_loss', cv_loss)
            print('cv_accuracy0', cv_accuracy0)
            print('cv_accuracy1', cv_accuracy1)
            torch.save(model.state_dict(), "Classifier.pth")
            print('==========')

        

    # 测试模式
    model.eval()
    test_loss = 0
    test_correct0 = 0
    test_correct1 = 0
    total_samples = 0
    with torch.no_grad():
        for data, labels in test_loader:
            # 将数据和标签转换为张量
            data = data.cuda()
            labels = labels.cuda()
    
            # 向前传递
            outputs = model(data)
            loss = new_loss(outputs, labels)

            splited_outputs0 = outputs[:, 0].unsqueeze(1)
            splited_outputs1 = outputs[:, 1:7]
    
            test_loss += loss.item() * data.size(0)
            predicted0 = (splited_outputs0 > 0.5).float()
            _, predicted1 = torch.max(splited_outputs1, 1)
            test_correct0 += (predicted0 == labels[:, 0].unsqueeze(1)).sum().item()
            test_correct1 += (predicted1 == labels[:, 1]).int().sum().item()
            total_samples += labels.size(0)
        test_loss /= len(train_loader.dataset)
        test_accuracy0 = test_correct0 / total_samples
        test_accuracy1 = test_correct1 / total_samples

        print('test_loss', test_loss)
        print('test_accuracy0', test_accuracy0)
        print('test_accuracy1', test_accuracy1)
        torch.save(model.state_dict(), "Classifier.pth")
        print('==========')
    
    # # 测试模式
    # model.eval()
    # test_loss = 0
    # test_correct = 0
    # with torch.no_grad():
    #     for data, labels in device_loader:
    #         # 将数据和标签转换为张量
    #         data = data.float()
    #         labels = labels.float()
    #
    #         # 向前传递
    #         outputs = model(data)
    #         loss = criterion(outputs, labels.unsqueeze(1))
    #
    #         test_loss += loss.item() * data.size(0)
    #         predicted = (outputs > 0.5).float()
    #         # print(predicted)
    #         test_correct += (predicted == labels.unsqueeze(1)).sum().item()
    #         test_loss /= len(test_loader.dataset)
    #         test_accuracy = test_correct / len(device_loader.dataset)
    # print('test_accuracy', test_accuracy)
    # torch.save(model.state_dict(), "speClassifier.pth")
    # print('below is device accuracy')