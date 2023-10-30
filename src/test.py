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


group_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# print(tensor_read)
# print(tensor_read.shape)

if __name__ == "__main__":

    testTrueAddress = 'vData/True'
    testFalseAddress = 'vData/False'
    trainTrueAddress = 'testData/True'
    trainFalseAddress = 'testData/False'

    # 读取TRUE数据
    # 指定文件夹路径
    folder_path = trainTrueAddress
    # folder_path = "/path/to/your/folder"

    # 获取文件夹下的所有子文件夹
    subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

    # 初始化一个空列表，用于存储每个CSV文件的数据
    data_list = []

    # 遍历每个子文件夹
    for subdir in subdirectories:
        subdir_path = os.path.join(folder_path, subdir)

        # 获取子文件夹下的CSV文件
        csv_files = [file for file in os.listdir(subdir_path) if file.endswith(".csv")]

        # 遍历每个CSV文件
        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)

            # 读取CSV文件数据
            csv_data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

            # 将数据添加到列表中
            data_list.append(csv_data)

    # 将数据列表转换为PyTorch张量
    # test_true_data = torch.tensor(data_list)
    true_data = data_list
    true_label = [1] * len(true_data)

    # 读取testTRUE数据
    # 指定文件夹路径
    folder_path = testTrueAddress
    # folder_path = "/path/to/your/folder"

    # 获取文件夹下的所有子文件夹
    subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

    # 初始化一个空列表，用于存储每个CSV文件的数据
    data_list = []

    # 遍历每个子文件夹
    for subdir in subdirectories:
        subdir_path = os.path.join(folder_path, subdir)

        # 获取子文件夹下的CSV文件
        csv_files = [file for file in os.listdir(subdir_path) if file.endswith(".csv")]

        # 遍历每个CSV文件
        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)

            # 读取CSV文件数据
            csv_data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

            # 将数据添加到列表中
            data_list.append(csv_data)

    # 将数据列表转换为PyTorch张量
    # test_true_data = torch.tensor(data_list)
    test_true_data = data_list
    test_true_label = [1] * len(test_true_data)

    # 读取FALSE数据
    # 指定文件夹路径
    folder_path = trainFalseAddress
    # folder_path = "/path/to/your/folder"

    # 获取文件夹下的所有子文件夹
    subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

    # 初始化一个空列表，用于存储每个CSV文件的数据
    data_list = []

    # 遍历每个子文件夹
    for subdir in subdirectories:
        subdir_path = os.path.join(folder_path, subdir)

        # 获取子文件夹下的CSV文件
        csv_files = [file for file in os.listdir(subdir_path) if file.endswith(".csv")]

        # 遍历每个CSV文件
        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)

            # 读取CSV文件数据
            csv_data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

            # 将数据添加到列表中
            data_list.append(csv_data)

    # 将数据列表转换为PyTorch张量
    # test_false_data = torch.tensor(data_list)
    false_data = data_list
    false_label = [0] * len(false_data)

    # 读取testFALSE数据
    # 指定文件夹路径
    folder_path = testFalseAddress
    # folder_path = "/path/to/your/folder"

    # 获取文件夹下的所有子文件夹
    subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]

    # 初始化一个空列表，用于存储每个CSV文件的数据
    data_list = []

    # 遍历每个子文件夹
    for subdir in subdirectories:
        subdir_path = os.path.join(folder_path, subdir)

        # 获取子文件夹下的CSV文件
        csv_files = [file for file in os.listdir(subdir_path) if file.endswith(".csv")]

        # 遍历每个CSV文件
        for csv_file in csv_files:
            csv_path = os.path.join(subdir_path, csv_file)

            # 读取CSV文件数据
            csv_data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)

            # 将数据添加到列表中
            data_list.append(csv_data)

    # 将数据列表转换为PyTorch张量
    # test_false_data = torch.tensor(data_list)
    test_false_data = data_list
    test_false_label = [0] * len(test_false_data)

    all_data = true_data + false_data
    all_label = true_label + false_label

    device_data = test_true_data + test_false_data
    device_label = test_true_label + test_false_label

    train_data, test_data, train_label, test_label = train_test_split(all_data, all_label, test_size=0.2,
                                                                      random_state=42, stratify=all_label)
    train_tensor = torch.tensor(train_data).to(device)
    test_tensor = torch.tensor(test_data).to(device)
    train_label_tensor = torch.tensor(train_label).to(device)
    test_label_tensor = torch.tensor(test_label).to(device)

    device_tensor = torch.tensor(device_data).to(device)
    device_label_tensor = torch.tensor(device_label).to(device)

    # train_tensor.to(device)
    # test_tensor.to(device)
    # train_label_tensor.to(device)
    # test_label_tensor.to(device)

    print('train=========', train_tensor.device)

    # 将数据和标签组成数据集
    train_dataset = TensorDataset(train_tensor, train_label_tensor)
    test_dataset = TensorDataset(test_tensor, test_label_tensor)
    device_dataset = TensorDataset(device_tensor, device_label_tensor)

    # 定义数据加载器
    batch_size = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    device_loader = DataLoader(device_dataset, batch_size=batch_size, shuffle=True)

    # with open(address+'/newWave'+str(index)+'-'+str(i)+'.csv', 'r') as file:
    # reader = csv.reader(file)
    # row = next(reader)
    # tensor_read = torch.from_numpy(np.array(row, dtype=np.float32))
    # 实例化模型
    model = BinaryClassifier()
    model = model.cuda()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    num_epochs = 120
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        train_correct = 0
        # print('train_loader.shape',train_loader.shape)
        for data, labels in train_loader:
            # 将数据和标签转换为张量
            data = data.float()
            labels = labels.float()

            # 向前传递
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            train_loss += loss.item() * data.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels.unsqueeze(1)).sum().item()

        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        print('train_loss', train_loss)
        print('train_accuracy', train_accuracy)

    # 测试模式
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            # 将数据和标签转换为张量
            data = data.float()
            labels = labels.float()

            # 向前传递
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1))

            test_loss += loss.item() * data.size(0)
            predicted = (outputs > 0.5).float()
            test_correct += (predicted == labels.unsqueeze(1)).sum().item()
            test_loss /= len(test_loader.dataset)
            test_accuracy = test_correct / len(test_loader.dataset)
    print('test_accuracy', test_accuracy)
    torch.save(model.state_dict(), "speClassifier.pth")
    print('==========')

    # 测试模式
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, labels in device_loader:
            # 将数据和标签转换为张量
            data = data.float()
            labels = labels.float()

            # 向前传递
            outputs = model(data)
            loss = criterion(outputs, labels.unsqueeze(1))

            test_loss += loss.item() * data.size(0)
            predicted = (outputs > 0.5).float()
            # print(predicted)
            test_correct += (predicted == labels.unsqueeze(1)).sum().item()
            test_loss /= len(test_loader.dataset)
            test_accuracy = test_correct / len(device_loader.dataset)
    print('test_accuracy', test_accuracy)
    torch.save(model.state_dict(), "speClassifier.pth")
    print('below is device accuracy')