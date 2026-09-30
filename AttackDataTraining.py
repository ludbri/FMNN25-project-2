import numpy as np
from FNNattack import attack
from NeuralNetwork import NeuralNetwork
from NeuralNetworkTraining import train_network, evaluate
import parameters


def generate_attack_data(network: NeuralNetwork,
                         training_data: tuple[np.ndarray, np.ndarray],
                         num_samples: int = 100,
                         target_strategy: str = 'random_sifferent'
                         ) -> tuple[np.ndarray, np.ndarray] :
    '''
    Generates attack samples from given dataset using trained samples.

    Parameters
    ----------
    network : NeuralNetwork
        Trained network to generate attack against
    training_data : tuple[np.ndarray, np.ndarray]
        Tuple of (images, labels).
    num_samples : int, optional
        Number of images from dataset to convert into attack samples
    target_strategy : str, default="random_different"
        How to pick the target class for the attack:
        - "random_different": Picks a random class index != true label.
        - "next_class": Targets (true_label + 1) % N_CLASSES.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (adv_images, true_labels) containing the generated adversarial examples 
        paired with their original ground-truth labels.
    '''
    
    images, labels = training_data
    n_total = len(images)
    num_samples = min(num_samples, n_total)
    
    adv_images = []
    adv_labels = []
    
    print(f"Generating {num_samples} adversarial examples using iterative `attack`...")
    
    for i in range(num_samples):
        img = images[i]
        true_label = int(labels[i])
        
        if target_strategy == 'random_different':
            possible_targets = [c for c in range(parameters.N_CLASSES) if c != true_label]
            target_class = int(np.random.choice(possible_targets))
            
        elif target_strategy == 'next_class':
            target_class = (true_label + 1) % parameters.N_CLASSES
        else:
            raise ValueError(f"Unknown target_strategy: {target_strategy}")
            
        # Generate adversarial example using the provided `attack` function
        # Note: attack() returns shape (1, input_size)
        perturbed_img = attack(network, img, target=target_class)
        
        # Squeeze to match dataset shape (input_size,)
        adv_images.append(perturbed_img.reshape(-1))
        
        # Ground-truth label remains the ORIGINAL true label so the network learns robustness
        adv_labels.append(true_label)

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"  Generated {i + 1}/{num_samples} adversarial samples.")

    return np.array(adv_images), np.array(adv_labels)

def attack_retrain(
        network: NeuralNetwork,
        training_data: tuple[np.ndarray, np.ndarray],
        validation_data: tuple[np.ndarray, np.ndarray],
        num_adv_samples: int = 200,
        epochs: int = 3,
        minibatch_size: int = 10,
        training_limit: int = 10000,
        validation_limit: int = 1000
    ) -> tuple[NeuralNetwork,dict]:
    
    '''
    Generates adversarial samples using `attack`, merges them into the 
    training dataset, and retrains the network.

    Parameters
    ----------
    network : NeuralNetwork
        Initial trained network model.
    training_data : tuple[np.ndarray, np.ndarray]
        Original training set (images, labels).
    validation_data : tuple[np.ndarray, np.ndarray]
        Validation set (images, labels).
    num_adv_samples : int, default=200
        Number of original samples to perturb for data augmentation.
    epochs : int, default=3
        Number of retraining epochs.
    minibatch_size : int, default=10
        Batch size for retraining.
    training_limit : int, default=10000
        Maximum training samples per epoch.
    validation_limit : int, default=1000
        Maximum validation samples to test per epoch.

    Returns
    -------
    tuple[NeuralNetwork, dict]
        Retrained network model and training history history dictionary.
    
    '''
    
    train_imgs, train_lbls = training_data
    
    # Generate adversarial examples
    adv_imgs, adv_lbls = generate_attack_data(
        network=network,
        training_data=training_data,
        num_samples=num_adv_samples,
        target_strategy="random_different"
    )

    # Augment training dataset with original + adversarial samples
    augmented_imgs = np.vstack([train_imgs, adv_imgs])
    augmented_lbls = np.concatenate([train_lbls, adv_lbls])

    # Shuffle the augmented dataset
    shuffle_idx = np.random.permutation(len(augmented_imgs))
    augmented_dataset = (augmented_imgs[shuffle_idx], augmented_lbls[shuffle_idx])

    print(f"\nRetraining dataset size: {len(augmented_imgs)} "
          f"({len(train_imgs)} original + {len(adv_imgs)} adversarial)")

    # Retrain the network on augmented dataset
    history = train_network(
        network=network,
        training_data=augmented_dataset,
        validation_data=validation_data,
        minibatch_size=minibatch_size,
        epochs=epochs,
        training_limit=min(training_limit, len(augmented_imgs)),
        validation_limit=validation_limit
    )

    return network, history 
        
if __name__ == "__main__":
    from dataloading import load_mnist  # Assuming your dataset loader function

    dataset_file = "mnist.pkl"

    print("Loading MNIST dataset...")
    training_data, validation_data, test_data = load_mnist(dataset_file)

    # Initialize and train initial network
    input_size = training_data[0].shape[1]
    output_size = 4 if parameters.BINARY_ENCODING else parameters.N_CLASSES
    
    net = NeuralNetwork(layer_sizes=(input_size, 64, output_size))
    
    print("--- Initial Training ---")
    train_network(net, training_data, validation_data, epochs=3)

    # Evaluate initial accuracy on clean data
    corr, total = evaluate(net, validation_data)
    print(f"Pre-retraining Validation Accuracy: {100.0 * corr / total:.2f}%")

    # Retrain network using adversarial data generated via `attack`
    print("\n--- Starting Adversarial Retraining ---")
    net, history = attack_retrain(
        network=net,
        training_data=training_data,
        validation_data=validation_data,
        num_adv_samples=100,
        epochs=3
    )

    # Final Evaluation
    corr, total = evaluate(net, validation_data)
    print(f"\nPost-retraining Validation Accuracy: {100.0 * corr / total:.2f}%")
    
    