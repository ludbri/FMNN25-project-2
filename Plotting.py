import matplotlib.pyplot as plt

import numpy as np
import parameters


# ============================================================
# Plot a minibatch
# ============================================================
# TODO present a minibatch (and predictions)


# ============================================================
# GRAPH 1: VALIDATION ACCURACY VS EPOCH
# ============================================================

def plot_validation_accuracy(history):
    plt.figure()
    plt.plot(history["epochs"], history["validation_accuracy"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.xticks(history["epochs"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_accuracy_vs_epoch.png", dpi=150)
    plt.show()


# ============================================================
# GRAPH 2: LOSS VS EPOCH
# ============================================================

def plot_loss(history):
    plt.figure()
    plt.plot(history["epochs"], history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss vs Epoch")
    plt.xticks(history["epochs"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_vs_epoch.png", dpi=150)
    plt.show()


# ============================================================
# GRAPH 3: MINI-BATCH SIZE VS VALIDATION ACCURACY
# ============================================================

def plot_batch_accuracy(results):
    batch_sizes = list(results.keys())
    accuracies = [results[b]["final_accuracy"] for b in batch_sizes]

    plt.figure()
    plt.plot(batch_sizes, accuracies, marker="o")
    plt.xlabel("Mini-batch size")
    plt.ylabel("Final Validation Accuracy (%)")
    plt.title("Mini-batch Size vs Validation Accuracy")
    plt.xticks(batch_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mini_batch_vs_accuracy.png", dpi=150)
    plt.show()


# ============================================================
# GRAPH 4: MINI-BATCH SIZE VS TRAINING TIME
# ============================================================

def plot_batch_time(results):
    batch_sizes = list(results.keys())
    times = [results[b]["time"] for b in batch_sizes]

    plt.figure()
    plt.plot(batch_sizes, times, marker="o")
    plt.xlabel("Mini-batch size")
    plt.ylabel("Training Time (seconds)")
    plt.title("Mini-batch Size vs Training Time")
    plt.xticks(batch_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mini_batch_vs_time.png", dpi=150)
    plt.show()


# ============================================================
# GRAPH 5: Confusion Matrix
# ============================================================

def confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix - y axis is the true class, x axis is the prediction. 
    Each row is normalized to 1.
    """
    counts = np.zeros((parameters.N_CLASSES, parameters.N_CLASSES), dtype=int)
    for y_t, y_p in zip(y_true, y_pred):
        counts[y_t,y_p] += 1
    counts /= counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()
    ax.matshow(counts, cmap='Greys', title="Confusion matrix")
    ticks = [range(10)]
    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)

    plt.show()

    return fig, ax

