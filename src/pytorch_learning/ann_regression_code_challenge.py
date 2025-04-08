from ann_regression import (
    data,
    model,
)
import numpy as np
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    n_epochs = 500
    slopes = np.linspace(-2, 2, 21)
    all_losses = np.zeros(len(slopes))
    all_corrcoef = np.zeros(len(slopes))
    print(slopes)

    for i, slope in enumerate(slopes):
        sum_loss = 0
        sum_corrcoef = 0
        n = 0
        for iter in range(50):
            x, y = data(line_slope=slope)
            y_ped, losses = model(x=x, y=y, n_epochs=n_epochs)
            sum_loss += losses[-1]
            cof = np.corrcoef(y.T, y_ped.T)[0, 1]
            if not math.isnan(cof):
                sum_corrcoef += np.corrcoef(y.T, y_ped.T)[0, 1]

        all_losses[i] = sum_loss / 50
        all_corrcoef[i] = sum_corrcoef / 50

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(slopes, all_losses, "o")
    ax[0].set_xlabel("slope")
    ax[0].set_ylabel("loss")

    ax[1].plot(slopes, all_corrcoef, "v")
    ax[1].set_xlabel("slope")
    ax[1].set_ylabel("Correlation Coef.")
    plt.show()
