import numpy as np
import matplotlib.pyplot as plt
import os

# --- MiniFlow Framework Imports ---
from miniflow.model.model import Model
from miniflow.layers.dense import Dense
from miniflow.layers.flatten import Flatten
from miniflow.activations import ReLU
from miniflow.losses.softmax_ce import SoftmaxCrossEntropy
from miniflow.optimizers.adam import Adam


# Setting print options for numpy arrays
np.set_printoptions(precision=4, suppress=True)


def load_mnist_data(path='mnist.npz'):
    """
    Loads the MNIST dataset from a local file and normalizes the data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File '{path}' not found.")

    print(f"📂 Loading dataset from {path}...")
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['train_data'], f['train_labels']
        x_test, y_test = f['test_data'], f['test_labels']

    # Normalizing pixels to the range 0 to 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"✅ Dataset Loaded: Train={x_train.shape}, Test={x_test.shape}")
    return x_train, y_train, x_test, y_test


def to_categorical(y, num_classes=10):
    """
    Converts labels to One-Hot Encoding format.
    Required for SoftmaxCrossEntropy.
    """
    if y.ndim > 1: return y
    return np.eye(num_classes)[y]


def plot_training_results(loss_values, x_test, y_test, preds, save_path='results.png'):
    """
    Plots the loss trend and displays predicted samples.
    (Revised version to avoid tight_layout error)
    """
    if not loss_values:
        print("⚠️ Loss list is empty, no plot will be drawn.")
        return

    # Create a figure with a suitable size (14 by 6 inches)
    plt.figure(figsize=(14, 6))

    # --- Section 1: Loss Plot (Left) ---
    # Using a 3-row, 6-column grid. The plot takes the first 3 columns.
    ax1 = plt.subplot2grid((3, 6), (0, 0), rowspan=3, colspan=3)
    ax1.plot(range(1, len(loss_values) + 1), loss_values, 'b-o', label='Training Loss', linewidth=2)
    ax1.set_title('Training Loss per Epoch', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- Section 2: Display Sample Images (Right) ---
    # Selecting 9 random images from the test data
    indices = np.random.choice(len(x_test), 9, replace=False)

    for i, idx in enumerate(indices):
        # Calculate position in the grid (3x3 block on the right)
        row = i // 3
        col = (i % 3) + 3  # +3 because the first 3 columns are for the main plot

        ax = plt.subplot2grid((3, 6), (row, col))

        # Handling image dimensions (if flattened, reshape back to 28x28)
        img = x_test[idx]
        if img.ndim == 1: img = img.reshape(28, 28)

        ax.imshow(img, cmap='gray')

        # Finding the predicted and true label
        pred_label = np.argmax(preds[idx])
        true_label = y_test[idx]

        # Green for correct, Red for incorrect
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"P:{pred_label} | T:{true_label}", color=color, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    print(f"📊 Results plot saved in file '{save_path}'.")
    plt.savefig(save_path)
    plt.show()


# --- Main Program Execution ---
if __name__ == "__main__":
    # 1. Data Preparation
    try:
        x_train, y_train, x_test, y_test = load_mnist_data()
    except Exception as e:
        print(e)
        exit()

    # Convert training labels to One-Hot (for the loss function)
    y_train_encoded = to_categorical(y_train)

    # 2. Model Construction
    print("\n🏗️ Building Model...")
    model = Model()

    # Input Layer: Converts 28x28 images into a 784-element vector
    model.add(Flatten(input_shape=(28, 28)))

    # First hidden layer
    model.add(Dense(128))
    model.add(ReLU())

    # Second hidden layer
    model.add(Dense(64))
    model.add(ReLU())

    # Output Layer: 10 neurons for 10 classes
    # Important Note: We do not use an Activation here because SoftmaxCrossEntropy
    # performs the Softmax operation on the Logits (raw output) itself.
    model.add(Dense(10))

    # 3. Compile Model
    # Using Adam optimizer and Softmax Cross Entropy loss function
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=SoftmaxCrossEntropy())

    # Initialize weights (a single dummy forward pass)
    model.predict(np.zeros((1, 28, 28)))
    model.summary()

    # 4. Model Training
    print("\n🚀 Starting Training...")
    # Train for 5 epochs
    history = model.fit(x_train, y_train_encoded, epochs=5, batch_size=64)

    # 5. Extract Loss List
    # Handling whether the fit output is a list or a History object
    loss_list = []
    if isinstance(history, list):
        loss_list = history
    elif hasattr(history, 'history') and 'loss' in history.history:
        loss_list = history.history['loss']
    elif hasattr(history, 'losses'):
        loss_list = history.losses
    else:
        # Last attempt to find the loss list in attributes
        print("⚠️ Warning: Could not auto-detect loss list format.")

    # 6. Evaluate on Test Data
    print("\n🧪 Evaluating on Test Data...")
    test_preds = model.predict(x_test)

    # Convert probabilities to labels (Argmax) and compare with true labels
    predictions_labels = np.argmax(test_preds, axis=1)
    accuracy = np.mean(predictions_labels == y_test)

    print(f"\n{'=' * 40}")
    print(f"🏆 Final Accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 40}\n")

    # 7. Plot and Save Results
    plot_training_results(loss_list, x_test, y_test, test_preds)
