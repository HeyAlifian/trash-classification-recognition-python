# ============================================================
# IMAGE CLASSIFICATION WITH A NEURAL NETWORK
# Built for high school learners — every line explained!
# ============================================================

# --- IMPORTS ---
# Think of imports like "loading tools from a toolbox"

import os                        # Helps us work with files and folders
import numpy as np               # "NumPy" — great for math and working with arrays (grids of numbers)
import matplotlib.pyplot as plt  # For drawing graphs and showing images

# TensorFlow & Keras — the main deep learning library we use to BUILD the neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# These help us prepare image data before feeding it to the neural network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

# ============================================================
# SECTION 1: CONFIGURATION
# These are settings you can easily change without touching the
# rest of the code.
# ============================================================

IMG_HEIGHT   = 128      # Each image will be resized to 128 pixels tall
IMG_WIDTH    = 128      # Each image will be resized to 128 pixels wide
BATCH_SIZE   = 32       # We train 32 images at a time (a "batch")
EPOCHS       = 20       # How many full passes through ALL training data
LEARNING_RATE = 0.001   # How big of a step the model takes when learning (small = careful)

DATASET_DIR = "datasets"   # Change this to wherever your images are stored

# ============================================================
# SECTION 2: LOAD THE DATA
# We read images from folders. Keras automatically uses the
# folder names as class labels (e.g. "cats", "dogs").
# ============================================================

def load_data(dataset_dir):
    """
    Loads training and validation image datasets from a folder.

    'Training data'   = images the model LEARNS from
    'Validation data' = images used to CHECK how well it learned
                        (model never trains on these — it's like a practice test!)
    """

    train_dir = os.path.join(dataset_dir, "train")  # Path: dataset/train
    val_dir   = os.path.join(dataset_dir, "val")    # Path: dataset/val

    # image_dataset_from_directory:
    #   - Reads all images from subfolders
    #   - Automatically assigns labels based on folder names
    #   - Resizes every image to our target size
    #   - Groups images into batches
    train_dataset = image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),  # Resize all images to the same size
        batch_size=BATCH_SIZE,
        label_mode="categorical",            # Labels become arrays like [1,0,0] for 3 classes
        shuffle=True,                        # Shuffle images so the model doesn't memorize order
        seed=42                              # "Seed" makes randomness repeatable (same shuffle every run)
    )

    val_dataset = image_dataset_from_directory(
        val_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,                       # No need to shuffle validation data
        seed=42
    )

    # Get class names from folder names automatically (e.g. ["cats", "dogs"])
    class_names = train_dataset.class_names
    print(f"\nFound {len(class_names)} classes: {class_names}")

    return train_dataset, val_dataset, class_names


# ============================================================
# SECTION 3: DATA AUGMENTATION
# Augmentation = making artificial variations of training images
# so the model sees more variety and doesn't just memorize.
#
# Example: flip a cat photo left-right → still a cat photo,
#          but the model sees it as "new" data!
# ============================================================

def build_augmentation_layer():
    """
    Returns a Sequential block of random image transformations.
    These are ONLY applied during training, not during validation.
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),        # 50% chance: flip image left-to-right
        layers.RandomRotation(0.1),             # Randomly rotate up to 10% of 360° (~36 degrees)
        layers.RandomZoom(0.1),                 # Randomly zoom in/out by up to 10%
        layers.RandomBrightness(0.1),           # Randomly tweak brightness ±10%
    ], name="data_augmentation")


# ============================================================
# SECTION 4: BUILD THE NEURAL NETWORK MODEL
#
# Our model is a CNN — Convolutional Neural Network.
# CNNs are specially designed for images.
#
# Here's the intuition:
#   Conv2D layers → "look" at small patches of the image to detect features
#                   (edges, colors, shapes, textures...)
#   MaxPooling    → shrink the image down (keep only the strongest signals)
#   Dense layers  → make the final decision based on all detected features
# ============================================================

def build_model(num_classes):
    """
    Builds and returns a CNN model.

    num_classes = how many categories to predict
                  (e.g. 3 if you have cats, dogs, birds)
    """

    augmentation = build_augmentation_layer()

    # keras.Sequential: layers stacked one after another, like a pipeline
    model = keras.Sequential([

        # --- INPUT LAYER ---
        # Tells the model what shape of data to expect
        # Shape: (height, width, 3 color channels: Red, Green, Blue)
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # --- AUGMENTATION (only active during training) ---
        augmentation,

        # --- NORMALIZATION ---
        # Pixel values range from 0–255. Divide by 255 to get 0.0–1.0.
        # Neural networks learn faster when numbers are small!
        layers.Rescaling(1.0 / 255),

        # ==============================
        # CONVOLUTIONAL BLOCK 1
        # ==============================
        # Conv2D: Scans 32 tiny 3×3 "filters" across the image
        # Each filter learns to detect something (e.g. a horizontal edge)
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        # ReLU activation: turns negative numbers to 0 (adds non-linearity — makes the model smarter)
        # padding="same": keeps the image the same size after convolution

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        # Another Conv layer: learns more complex patterns from the first layer

        layers.MaxPooling2D(pool_size=(2, 2)),
        # MaxPooling: takes each 2×2 block of pixels, keeps only the MAX value
        # → cuts image dimensions in half, reduces computation

        layers.Dropout(0.25),
        # Dropout: randomly "turns off" 25% of neurons during training
        # Forces the network to not rely too heavily on any single neuron
        # → prevents OVERFITTING (memorizing instead of learning)

        # ==============================
        # CONVOLUTIONAL BLOCK 2
        # ==============================
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        # 64 filters now — we look for more complex patterns at this stage

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0.25),

        # ==============================
        # CONVOLUTIONAL BLOCK 3
        # ==============================
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        # 128 filters — even more complex patterns

        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0.25),

        # ==============================
        # TRANSITION: 2D → 1D
        # ==============================
        layers.Flatten(),
        # Flatten: takes the 2D feature map and stretches it into a 1D list
        # Like unrolling a grid into a single long row of numbers

        # ==============================
        # FULLY CONNECTED (DENSE) LAYERS
        # ==============================
        layers.Dense(256, activation="relu"),
        # Dense: every neuron connects to every neuron in the previous layer
        # 256 neurons = 256 "opinions" about what the image might be

        layers.Dropout(0.5),
        # Higher dropout here (50%) — Dense layers are prone to overfitting

        # ==============================
        # OUTPUT LAYER
        # ==============================
        layers.Dense(num_classes, activation="softmax"),
        # num_classes neurons — one for each category
        # Softmax: converts outputs into probabilities that sum to 1.0
        # e.g. [0.85, 0.10, 0.05] → 85% cat, 10% dog, 5% bird
    ])

    return model


# ============================================================
# SECTION 5: COMPILE THE MODEL
# Compilation = telling the model HOW to learn
#
# - Optimizer: the algorithm that updates weights (Adam is popular & reliable)
# - Loss function: measures how WRONG the model is
#   (categorical_crossentropy is standard for multi-class classification)
# - Metrics: what we want to track while training (accuracy!)
# ============================================================

def compile_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ============================================================
# SECTION 6: CALLBACKS
# Callbacks = special actions that happen automatically during training
# ============================================================

def get_callbacks():
    callbacks = [

        # ModelCheckpoint: saves the model whenever validation accuracy improves
        # → you keep the BEST version, not the last version
        keras.callbacks.ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",    # Watch validation accuracy
            save_best_only=True,       # Only save if it's better than before
            verbose=1                  # Print a message when it saves
        ),

        # EarlyStopping: stops training if the model stops improving
        # → prevents wasting time and overfitting
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,        # Stop if no improvement for 5 consecutive epochs
            restore_best_weights=True,  # Go back to the best weights when stopping
            verbose=1
        ),

        # ReduceLROnPlateau: lowers learning rate when progress slows down
        # → lets the model take smaller, more careful steps to fine-tune
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,        # New LR = current LR × 0.5 (halved)
            patience=3,        # Trigger after 3 epochs with no improvement
            min_lr=1e-6,       # Never go below this learning rate
            verbose=1
        ),
    ]
    return callbacks


# ============================================================
# SECTION 7: VISUALIZE TRAINING HISTORY
# After training, we plot graphs to understand what happened
# ============================================================

def plot_training_history(history):
    """
    'history' is the object returned by model.fit().
    It records accuracy and loss for every epoch.
    """

    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Accuracy ---
    axes[0].plot(epochs_range, acc,     label="Training Accuracy",   color="steelblue")
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="coral",    linestyle="--")
    axes[0].set_title("Model Accuracy Over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # TIP: If training accuracy >> validation accuracy → OVERFITTING
    #      The model memorized training data but can't generalize

    # --- Plot 2: Loss ---
    axes[1].plot(epochs_range, loss,     label="Training Loss",   color="steelblue")
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="coral",    linestyle="--")
    axes[1].set_title("Model Loss Over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # TIP: Loss should go DOWN over time. If val_loss starts rising → overfitting

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Training graph saved as 'training_history.png'")


# ============================================================
# SECTION 8: PREDICT ON A SINGLE IMAGE
# After training, use the model to classify a new image
# ============================================================

def predict_image(model, image_path, class_names):
    """
    Load a single image and have the model predict its class.
    """
    # Load & resize the image to match our training size
    img = keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Convert image to a NumPy array of pixel values
    img_array = keras.utils.img_to_array(img)

    # The model expects a BATCH of images, not just one.
    # np.expand_dims adds an extra dimension: shape (H, W, 3) → (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # model.predict() returns probabilities for each class
    predictions = model.predict(img_array, verbose=0)  # shape: (1, num_classes)
    predictions = predictions[0]                       # grab the single result

    # np.argmax finds the index of the highest probability
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence      = predictions[predicted_index] * 100

    print(f"\n>>> Prediction for: {image_path}")
    print(f"   - Predicted class : {predicted_class}")
    print(f"   - Confidence      : {confidence:.1f}%")
    print(f"   - All probabilities:")
    for name, prob in zip(class_names, predictions):
        bar = "█" * int(prob * 20)
        print(f"      {name:<15} {prob*100:5.1f}%  {bar}")

    return predicted_class, confidence


# ============================================================
# SECTION 9: MAIN — TIE EVERYTHING TOGETHER
# This is where the full training pipeline runs
# ============================================================

def main():
    # --- Step 1: Load data ---
    print("\n>>> Loading dataset...")
    train_ds, val_ds, class_names = load_data(DATASET_DIR)
    num_classes = len(class_names)

    # --- Step 2: Optimize data loading speed ---
    # AUTOTUNE lets TensorFlow decide the best number of parallel threads
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # cache()    → stores images in memory after first load (faster subsequent epochs)
    # prefetch() → loads next batch while the model is training on the current one
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # --- Step 3: Build model ---
    print("\n>>> Building the neural network...")
    model = build_model(num_classes)
    model = compile_model(model)

    # Print a summary: shows each layer, its output shape, and parameter count
    model.summary()

    # --- Step 4: Train ---
    print(f"\n>>> Training for up to {EPOCHS} epochs...")
    print(f"   Classes  : {class_names}")
    print(f"   Image size: {IMG_HEIGHT}×{IMG_WIDTH}")
    print(f"   Batch size: {BATCH_SIZE}\n")

    history = model.fit(
        train_ds,           
        epochs=EPOCHS,  
        validation_data=val_ds,
        callbacks=get_callbacks()
    )

    print("\n>>> Final evaluation...")
    val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"   Validation Loss    : {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_accuracy * 100:.2f}%")

    plot_training_history(history)

    model.save("model_final.keras")
    print("\n>>> Model saved as 'model_final.keras'")
    print("   (Load it later with: model = keras.models.load_model('model_final.keras'))")

if __name__ == "__main__":
    main()