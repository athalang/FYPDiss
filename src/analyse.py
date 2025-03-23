import matplotlib.pyplot as plt
import numpy as np

def loss_distribution(losses, ax):
    ax.hist(losses, bins=30, alpha=0.7)
    ax.set_title("Loss Distribution Over Epochs")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Frequency")
    ax.grid(True)

def loss_over_time(train, val, ax):
    ax.plot(train, label="Train")
    ax.plot(val, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss Across Epochs")
    ax.grid(True)

def loss_delta(losses, ax):
    delta = np.diff(losses)
    ax.plot(delta)
    ax.set_title("Change in Loss Per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ Loss")
    ax.grid(True)

def show(train, val):
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), tight_layout=True)

    loss_over_time(train, val, axs[0])
    loss_distribution(val, axs[1])
    loss_delta(val, axs[2])

    plt.show()