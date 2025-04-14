import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader, TensorDataset

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def create_data():
    n_per_cluster = 200

    theta = np.linspace(0, 2 * np.pi, n_per_cluster)
    r1 = 10
    r2 = 15

    (x1, y1) = (
        r1 * np.cos(theta) + np.random.randn(n_per_cluster) * 3,
        r1 * np.sin(theta) + np.random.randn(n_per_cluster),
    )

    (x2, y2) = (
        r2 * np.cos(theta) + np.random.randn(n_per_cluster),
        r2 * np.sin(theta) + np.random.randn(n_per_cluster) * 3,
    )

    label = np.append(np.zeros(n_per_cluster), np.ones(n_per_cluster))

    x = np.append(x1, x2)
    y = np.append(y1, y2)

    data = np.column_stack((x, y))

    return data, np.reshape(label, (len(label), 1))


class ANNModel(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.input = nn.Linear(2, 128, dtype=float)
        self.hidden_layer = nn.Linear(128, 128, dtype=float)
        self.output = nn.Linear(128, 1, dtype=float)

        self.training = True
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = F.relu(self.hidden_layer(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.output(x)

        return x


def create_model(dropoutrate):
    model = ANNModel(dropout_rate=dropoutrate)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.002)

    return model, loss_function, optimizer


def train_model(model, loss_func, optimizer, train_loader, test_loader):
    n_epochs = 1000

    train_accuracy = []
    test_accuracy = []

    for epoch_i in range(n_epochs):

        model.training = True
        batch_accuracy = []

        for X, y in train_loader:
            y_hat = model(X)

            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_accuracy.append(100 * torch.mean(((y_hat > 0) == y).float()).item())

        train_accuracy.append(np.mean(batch_accuracy))

        model.training = False
        X, y = next(iter(test_loader))
        y_hat = model(X)
        test_accuracy.append(100 * torch.mean(((y_hat > 0) == y).float()).item())

    return train_accuracy, test_accuracy


if __name__ == "__main__":
    data, label = create_data()
    if False:
        print(data)
        data1 = data[label[:, 0] == 0]
        data2 = data[label[:, 0] == 1]

        plt.plot(data1[:, 0], data1[:, 1], "o")
        plt.plot(data2[:, 0], data2[:, 1], "v")

        plt.show()
    train_data, test_data, train_label, test_label = train_test_split(
        data, label, test_size=0.2
    )

    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
    test_dataset = TensorDataset(torch.tensor(test_data), torch.tensor(test_label))

    train_loader = dataloader.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = dataloader.DataLoader(
        test_dataset, batch_size=test_dataset.tensors[0].shape[0]
    )

    model, loss_function, optimizer = create_model(dropoutrate=0.25)

    train_accuracy, test_accuracy = train_model(
        model, loss_function, optimizer, train_loader, test_loader
    )

    plt.plot(list(range(1000)), train_accuracy, "o")
    plt.plot(list(range(1000)), test_accuracy, "v")

    plt.show()
