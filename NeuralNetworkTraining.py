import numpy as np
import tqdm
from dataloading import minibatches, Dataset
from NeuralNetwork import NeuralNetwork


def evaluate(network: NeuralNetwork,
             dataset: Dataset,
             limit: int = None)-> tuple[int, int]:
    """
    Evaluates a network's prediction accuracy on a dataset.

    Parameters
    ----------
    network : NeuralNetwork
        The trained (or in-training) network to evaluate.
    dataset : Dataset
        A (images, y_true) pair -- input samples and their true labels.
    limit : int, optional
        Maximum number of samples to evaluate. If None, evaluates on
        the entire dataset. If greater than the dataset size, it is
        capped to the dataset size.

    Returns
    -------
    tuple[int, int]
        (number_correct, number_tested) -- the count of correctly
        classified samples and the total number of samples evaluated.
    """
    images, y_true = dataset
    
    # Cap the evaluation to `limit` samples, or use the full dataset if none given
    if limit is None:
        limit = len(images)
    else:
        limit = min(limit, len(images))
        
    # Predict on the first `limit` samples and count matches against true labels
    y_pred = network.predict(images[:limit])
    correct = np.sum(np.equal(y_pred, y_true[:limit]))

    return correct, limit

HistoryDict = dict[str,list[int|float]]

def train_network(network: NeuralNetwork,
                  training_data: Dataset,
                  validation_data: Dataset,
                  minibatch_size: int =10,
                  epochs: int =3,
                  training_limit: int = 10000, 
                  validation_limit: int = 1000) -> HistoryDict:
    """
    Trains a network for a given number of epochs using mini-batch
    gradient descent, tracking loss and validation accuracy per completed epoch.

    Parameters
    ----------
    network : NeuralNetwork
        The network to train.
    training_data : Dataset
        Dataset used for training.
    validation_data : Dataset
        Dataset used to measure validation accuracy after each epoch.
    minibatch_size : int, default=10
        Number of samples per mini-batch.
    epochs : int, default=3
        Number of full passes over the training data (capped at
        `training_limit` samples per epoch).
    training_limit : int, default=10000
        Maximum number of training samples to use per epoch. Capped to
        the size of `training_data` if smaller.
    validation_limit : int, default=1000
        Maximum number of validation samples to evaluate on. Capped to
        the size of `validation_data` if smaller.

    Returns
    -------
    HistoryDict
        Dictionary with keys "epochs", "training_loss", "validation_loss", 
            and "validation_accuracy",
        each a list with one entry per completed epoch, suitable for
        passing to the plotting functions.
    """
    # Save metrics for plots
    training_limit = min(training_limit, len(training_data[0]))
    validation_limit = min(validation_limit, len(validation_data[0]))

    history = {
        "epochs": [],
        "training_loss": [],
        "validation_loss": [],
        "validation_accuracy": []
    }
    
    # Number of batches expected per epoch, for the tqdm progress bar total
    batches_per_epoch = int(np.ceil(training_limit / minibatch_size))

    for epoch in tqdm.trange(epochs, desc="Training epochs"):
        batches = 0
        
        # Iterate over shuffled, one-hot-encoded mini-batches for this epoch
        for x, y_onehot in tqdm.tqdm(minibatches(training_data, 
                                                 batch_size=minibatch_size, 
                                                 n=training_limit, 
                                                 one_hot=True, 
                                                 shuffle=True),
                                     desc="batches",
                                     total= batches_per_epoch,
                                     leave=False):
            network.learn_batch(x, y_onehot, epoch_count = epoch)

            batches += 1

        # evaluate loss on all data
        
            
        # evaluate loss on all data
        for x, y_onehot in minibatches(training_data, batch_size=training_limit, n=training_limit):
            training_loss = network.evaluate_loss(x, y_onehot)

        for x, y_onehot in minibatches(validation_data, batch_size=validation_limit, n=validation_limit):
            validation_loss = network.evaluate_loss(x, y_onehot)
        
        # Evaluate on the test and validation sets after each epoch
        correct, total = evaluate(
            network,
            validation_data,
            validation_limit
        )
        validation_accuracy = 100.0 * correct / total

        history["epochs"].append(epoch + 1)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        #print(f"Epoch {epoch + 1}/{epochs} completed.")
        #print(f"  Loss: {average_loss:.6f}")
        #print(f"  Validation accuracy: {validation_accuracy:.2f}%")
        #print()

    return history


