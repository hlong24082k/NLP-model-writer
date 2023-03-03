import os
import tqdm
import argparse

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data import TextLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/archive', help='import data path')
    parser.add_argument('--', type=str, default='data/archive', help='import data path')
    

def train_model(model, train_data, optimizer, criterion, epoches, batch_size, device):
    X_train, Y_train = train_data
    # TextLoader = TextLoader()
    # X_train = TextLoader.token2id(X_train, TextLoader._vocal_size)
    X_train, y_train = torch.tensor(X_train), torch.tensor(Y_train).to(float)
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True) 

    acc = 0
    losses = []
    for epoch in tqdm(range(epoches)):
        for i, (datapoints, labels) in enumerate(train_loader):
            datapoints = datapoints.long().to(device)
            labels = labels.float().to(device).reshape(-1,1)

            preds = model(datapoints)
            optimizer.zero_grad()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
        print(f"\nLoss: {loss.detach().cpu().item()}, epoch: {epoch}/{epoches}")
        losses.append(loss.detach().cpu().item())
    return model

def train(args):
    data_loader = TextLoader(args.data_dir, args.vocab_size)
    train_data, test_data = data_loader.split_data()