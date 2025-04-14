import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == "__main__":
    x = torch.ones(5)

    dropout = nn.Dropout(0.2)
    y = dropout(x)

    print(y)
    print(y.mean())

    dropout.eval()

    y = dropout(x)

    print(y)

    dropout.train()

    y = dropout(x)

    print(y)

    y = F.dropout(x, training=False)

    print(y)
