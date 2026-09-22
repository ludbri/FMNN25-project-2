import numpy as np
import tqdm
from dataloading import minibatches, Dataset
from NeuralNetwork import NeuralNetwork


def evaluate(network: NeuralNetwork,
             dataset: Dataset,
             limit: int = None)-> tuple[int, int]:
    """
    TODO: documentation
    Return (number_correct, number_tested).
    """
    images, y_true = dataset
    if limit is None:
        limit = len(images)
    else:
        limit = min(limit, len(images))

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
    TODO: documentation
    """
    # Save metrics for plots
    training_limit = min(training_limit, len(training_data[0]))
    validation_limit = min(validation_limit, len(validation_data[0]))

    history = {
        "epochs": [],
        "loss": [],
        "validation_accuracy": []
    }
    global_step = 0

    batches_per_epoch = training_limit//minibatch_size
    if batches_per_epoch != int(batches_per_epoch): batches_per_epoch += 1

    for epoch in tqdm.trange(epochs, desc="Training epochs"):
        total_loss = 0.0
        batches = 0

        for x, y_onehot in tqdm.tqdm(minibatches(training_data, 
                                                 batch_size=minibatch_size, 
                                                 n=training_limit, 
                                                 one_hot=True, 
                                                 shuffle=True),
                                     desc="batches",
                                     total= batches_per_epoch,
                                     leave=False):
            total_loss += network.train_batch(x, y_onehot, epoch_count = epoch)

            global_step += 1
            batches += 1

        average_loss = total_loss / batches

        correct, total = evaluate(
            network,
            validation_data,
            validation_limit
        )
        validation_accuracy = 100.0 * correct / total

        history["epochs"].append(epoch + 1)
        history["loss"].append(average_loss)
        history["validation_accuracy"].append(validation_accuracy)

        #print(f"Epoch {epoch + 1}/{epochs} completed.")
        #print(f"  Loss: {average_loss:.6f}")
        #print(f"  Validation accuracy: {validation_accuracy:.2f}%")
        #print()

    return history


