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
    """
    Plots validation accuracy over training epochs and saves the figure
    to disk as a PNG.

    Parameters
    ----------
    history : dict
        Dictionary with keys "epochs" (list/array of epoch numbers) and
        "validation_accuracy" (list/array of validation accuracy values,
        as percentages, one per epoch).

    Returns
    -------
    None
        Displays the plot and saves it to "validation_accuracy_vs_epoch.png".
    """
    
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
    """
    Plots training loss (mean squared error) over training epochs and
    saves the figure to disk as a PNG.

    Parameters
    ----------
    history : dict
        Dictionary with keys "epochs" (list/array of epoch numbers) and
        "loss" (list/array of MSE values, one per epoch).

    Returns
    -------
    None
        Displays the plot and saves it to "training_loss_vs_epoch.png".
    """
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
    """
    Plots final validation accuracy against mini-batch size and saves
    the figure to disk as a PNG.

    Parameters
    ----------
    results : dict
        Dictionary keyed by mini-batch size, where each value is a dict
        containing at least a "final_accuracy" key (validation accuracy,
        as a percentage, achieved at that batch size).

    Returns
    -------
    None
        Displays the plot and saves it to "mini_batch_vs_accuracy.png".
    """
    # Extract batch sizes and their corresponding final accuracies, in matching order
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
    """
    Plots training time against mini-batch size and saves the figure
    to disk as a PNG.

    Parameters
    ----------
    results : dict
        Dictionary keyed by mini-batch size, where each value is a dict
        containing at least a "time" key (training time in seconds for
        that batch size).

    Returns
    -------
    None
        Displays the plot and saves it to "mini_batch_vs_time.png".
    """
    # Extract batch sizes and their corresponding training times, in matching order
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
    Computes and plots the confusion matrix as a greyscale heatmap.
    Rows are true classes, columns are predicted classes. Each row is
    normalized to sum to 1.

    Parameters
    ----------
    y_true : array-like
        True class labels (integers), one per sample.
    y_pred : array-like
        Predicted class labels (integers), one per sample.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the plotted confusion matrix.
    """
    # Tally counts[true_class, predicted_class] for every sample
    counts = np.zeros((parameters.N_CLASSES, parameters.N_CLASSES), dtype=float)
    
    # Normalize each row (true class) to sum to 1
    for y_t, y_p in zip(y_true, y_pred):
        counts[y_t,y_p] += 1
        
    counts /= counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()
    ax.matshow(counts, cmap='Greys', title="Confusion matrix")
    ax.set_title("Confusion matrix")
    ticks = list(range(10))
    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)

    plt.show()

    return fig, ax

