import tensorflow as tf
import numpy as np
from utils.data_utils import load_images, preprocess_images
from config import CNN_MODEL_PATH


def build_cnn_model(input_shape):
    """Build CNN architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_save_cnn_model():
    # Load and preprocess data
    originals, counterfeits = load_images()
    X = preprocess_images(originals + counterfeits)
    X = np.expand_dims(X, axis=-1)  # Add channel dimension
    y = np.array([0] * len(originals) + [1] * len(counterfeits))

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Build and train model
    model = build_cnn_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save model
    model.save(CNN_MODEL_PATH)
    print(f"CNN model saved to {CNN_MODEL_PATH}")