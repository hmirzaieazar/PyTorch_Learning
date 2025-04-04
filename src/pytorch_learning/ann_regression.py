import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def data(line_slope=1):
    n_data = 50
    x = torch.randn(n_data)
    y = line_slope * x + torch.randn(n_data) / 2

    return x.reshape(n_data, 1), y.reshape(n_data, 1)


def model(x, y, n_epochs):
    losses = np.zeros(n_epochs)

    ann_regression = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 1),
    )
    print(ann_regression.parameters())
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(ann_regression.parameters(), lr=0.05)

    for i_epoch in range(n_epochs):
        y_hat = ann_regression(x)

        loss = loss_func(y, y_hat)
        losses[i_epoch] = loss.detach()

        optimizer.zero_grad()  # reset the gradients of all optimized (torch.tensor)s
        loss.backward()  # updates the gradients by back propagation
        optimizer.step()  # updates the parameters

    y_pred = ann_regression(x).detach()
    return y_pred, losses


if __name__ == "__main__":
    n_epochs = 500

    x, y = data()

    y_pred, losses = model(x=x, y=y, n_epochs=n_epochs)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(x, y, "o")
    ax[0].plot(x, y_pred, "v")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[1].plot(list(range(len(losses))), losses)
    ax[1].set_xlabel("n_epoch")
    ax[1].set_ylabel("loss")

    plt.show()
