import pickle
from pathlib import Path
import numpy as np
from typing import Generator
import parameters

Dataset = tuple[np.array, np.array]

def load_mnist(filename: str | Path) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load the extracted supplied mnist.pkl.gz file.    
    Each split has the form: (images, labels).

    Assumes that the provided file is already extracted from the .gz format.
    """
    with open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


Minibatch = tuple[np.array, np.array]

def minibatches(dataset: Dataset,
                batch_size : int = None,
                n: int = None,
                one_hot: bool = True,
                shuffle: bool = False
                ) -> Generator[Minibatch, None, None]:
    """
    A generator that yields minibatches from a dataset.
    A minibatch is a 2-tuple of np.arrays: the x-data and y-data, where the y-data may be one-hot encoded.

    dataset:
        the dataset to iterate over, with same structure as a minibatch.

    batch_size:
        the number of samples per minibatch.
        If None, yields a single minibatch of all (x,y) samples.
    
    n:
        the maximum total number of samples to provide.
        If None, yields all samples in the dataset.
        Selects the samples before shuffling.

    one_hot:
        If True, the labels (y) are returned as a one-hot encoding.

    shuffle:
        If True, the dataset is shuffled before yielding.
    """
    x, y = dataset[0], dataset[1]
    if n is not None:
        x, y = x[:n], y[:n]
    if batch_size is None:
        batch_size = len(x)
    if shuffle:
        idxs = np.arange(len(x))
        np.random.shuffle(idxs)
        x = x[idxs,:]
        y = y[idxs]

    for start in range(0, n, batch_size):
        end = min(n, start+batch_size)
        size = end-start
        x_batch = x[start:end]
        y_batch = y[start:end]
        if one_hot:
            y_onehot = np.zeros((size, parameters.N_CLASSES), dtype='d')
            y_onehot[np.arange(size), y_batch] = 1.
            yield (x_batch, y_onehot)

        else:
            yield (x_batch, y_batch)


def test_minibatches():
    """
    Sanity-checks `minibatches` against a 22-sample slice of the MNIST
    training set with batch_size=5.

    Expects 5 batches total: four full batches of 5 samples, followed by
    one partial batch of the 2 remaining samples (22 = 4*5 + 2). Checks
    that each batch's labels are one-hot encoded with width
    `parameters.N_CLASSES`, and that batch sizes match this expected
    4-full-plus-1-partial pattern.

    """
    fp = Path("mnist.pkl")
    train, val, test = load_mnist(fp)

    i = 0
    for x,y in minibatches(train, 5, 22):
        assert y.shape[1] == parameters.N_CLASSES
        i += 1
        if i < 5: 
            assert y.shape[0] == 5
        else:
            assert y.shape[0] == 2

    assert i == 5


if __name__ == "__main__":
    fp = Path("mnist.pkl")

    train, val, test = load_mnist(fp)
    
    # Demo: print each batch's first image and its shape, plus the
    # corresponding one-hot labels, with shuffling disabled so output
    # is deterministic/inspectable
    for x, y in minibatches(train,
                          batch_size=5,
                          n = 22,
                          one_hot=True,
                          shuffle=False):
        print(f"x: {x[0]}, shape: {x.shape}")
        print(f"y: {y}")

    test_minibatches()
