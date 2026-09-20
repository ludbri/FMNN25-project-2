import random
    
    
# def load_mnist(filename):
#     """
#     Load the supplied mnist.pkl.gz file.
#     Each split has the form: (images, labels).
#     """
#     with gzip.open(filename, "rb") as f:
#         training_data, validation_data, test_data = pickle.load(
#             f, encoding="latin1"
#         )
#     return training_data, validation_data, test_data

 
def one_hot(label):
    """Convert digit 0..9 to a 10-element one-hot vector."""
    target = [0.0] * 10
    target[int(label)] = 1.0
    return target


def evaluate(network, dataset, limit=None):
    """Return (number_correct, number_tested)."""
    images, labels = dataset
    if limit is None:
        limit = len(images)
    else:
        limit = min(limit, len(images))

    correct = 0
    for index in range(limit):
        image = images[index]
        label = int(labels[index])
        if network.predict(image) == label:
            correct += 1

    return correct, limit


def train_network(network, training_data, validation_data,
                  mini_batch_size=10, epochs=3,
                  training_limit=10000, validation_limit=1000):
    # Save metrics for plots
    images, labels = training_data
    training_limit = min(training_limit, len(images))
    validation_limit = min(validation_limit, len(validation_data[0]))

    indices = list(range(training_limit))

    history = {
        "epochs": [],
        "loss": [],
        "validation_accuracy": []
    }
    global_step = 0

    for epoch in range(epochs):
        random.shuffle(indices)

        total_loss = 0.0
        batches = 0

        for start in range(0, training_limit, mini_batch_size):
            batch_indices = indices[start:start + mini_batch_size]

            mini_batch = [
                (images[i], one_hot(labels[i]))
                for i in batch_indices
            ]

            total_loss += network.train(mini_batch, j = epoch) # j resets after each epoch, is that what we want?
            global_step += 1
            batches += 1

            processed = min(start + mini_batch_size, training_limit)
            if processed % 1000 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: {processed}/{training_limit} examples")

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

        print(f"Epoch {epoch + 1}/{epochs} completed.")
        print(f"  Loss: {average_loss:.6f}")
        print(f"  Validation accuracy: {validation_accuracy:.2f}%")
        print()

    return history


