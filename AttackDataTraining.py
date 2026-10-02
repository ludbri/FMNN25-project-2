import numpy as np
from FNNattack import attack, attack_negative_gradient
from NeuralNetwork import NeuralNetwork
from NeuralNetworkTraining import train_network, evaluate
import parameters


def generate_attack_data(
    network: NeuralNetwork,
    training_data: tuple[np.ndarray, np.ndarray],
    num_samples: int = 100,
    attack_type: str = "target_attack",
    target_strategy: str = "random_different"
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Generates attack samples from a given dataset using a trained network.

    Parameters
    ----------
    network : NeuralNetwork
        Trained network to generate attack against.
    training_data : tuple[np.ndarray, np.ndarray]
        Tuple of (images, labels).
    num_samples : int, optional
        Number of images from dataset to convert into attack samples.
    attack_type : str, default="target_attack"
        The attack method to use:
        - "target_attack": Targeted attack toward a specific target class.
        - "negative_gradient": Untargeted attack along the negative gradient.
    target_strategy : str, default="random_different"
        How to pick the target class for targeted attacks (ignored for "negative_gradient"):
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
    
    print(f"Generating {num_samples} adversarial examples using `{attack_type}`...")
    
    for i in range(num_samples):
        img = images[i]
        true_label = int(labels[i])
        
        if attack_type == "target_attack":
            if target_strategy == 'random_different':
                possible_targets = [c for c in range(parameters.N_CLASSES) if c != true_label]
                target_class = int(np.random.choice(possible_targets))
            elif target_strategy == 'next_class':
                target_class = (true_label + 1) % parameters.N_CLASSES
            else:
                raise ValueError(f"Unknown target_strategy: {target_strategy}")
                
            perturbed_img = attack(network, img, target=target_class)
            
        elif attack_type == "negative_gradient":
            perturbed_img = attack_negative_gradient(network, img, step_size=0.01)
            
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")
            
        # Squeeze to match dataset shape (input_size,)
        adv_images.append(perturbed_img.reshape(-1))
        
        # Ground-truth label remains the original label for adversarial training
        adv_labels.append(true_label)

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            print(f"  Generated {i + 1}/{num_samples} adversarial samples.")

    return np.array(adv_images), np.array(adv_labels)

def attack_retrain(
        network: NeuralNetwork,
        training_data: tuple[np.ndarray, np.ndarray],
        validation_data: tuple[np.ndarray, np.ndarray],
        att: str,
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
        attack_type = att
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
        

    
    