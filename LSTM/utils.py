import tqdm
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


def train(self, model, device, train_loader, optimizer, 
            criterion, epoch):
    acc = 0
    epoches = 25

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
        print(f"\n{loss.detach().cpu().item()}")
        losses.append(loss.detach().cpu().item())

def visulize_loss(losses):
    total_losses = losses
    fig, ax = plt.subplots()
    ax.plot(total_losses)
    ax.set_title('Total Loss over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    plt.show()

