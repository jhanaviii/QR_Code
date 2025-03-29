import joblib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils.data_utils import load_images, preprocess_images
from utils.feature_utils import extract_all_features
from config import ML_MODEL_PATH, CNN_MODEL_PATH
import os
from datetime import datetime

# Create results directory if it doesn't exist
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_evaluation_report(report, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_report_{timestamp}.txt"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(report)
    print(f"Saved {model_name} report to {filepath}")


def save_confusion_matrix(cm, model_name, display_labels):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_cm_{timestamp}.png"
    filepath = os.path.join(RESULTS_DIR, filename)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved {model_name} confusion matrix to {filepath}")


def evaluate_ml_model():
    # Load model and data
    model = joblib.load(ML_MODEL_PATH)
    originals, counterfeits = load_images()

    # Prepare features
    X = extract_all_features(originals + counterfeits)
    y = np.array([0] * len(originals) + [1] * len(counterfeits))

    # Predict and evaluate
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=['Original', 'Counterfeit'])

    print("ML Model Classification Report:")
    print(report)
    save_evaluation_report(report, 'random_forest')

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    save_confusion_matrix(cm, 'random_forest', ['Original', 'Counterfeit'])


def evaluate_cnn_model():
    # Load model and data
    model = tf.keras.models.load_model(CNN_MODEL_PATH)
    originals, counterfeits = load_images()

    # Preprocess images
    X = preprocess_images(originals + counterfeits)
    X = np.expand_dims(X, axis=-1)
    y = np.array([0] * len(originals) + [1] * len(counterfeits))

    # Predict and evaluate
    y_pred = (model.predict(X) > 0.5).astype(int)
    report = classification_report(y, y_pred, target_names=['Original', 'Counterfeit'])

    print("\nCNN Model Classification Report:")
    print(report)
    save_evaluation_report(report, 'cnn')

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    save_confusion_matrix(cm, 'cnn', ['Original', 'Counterfeit'])

    # Save prediction probabilities for analysis
    y_probs = model.predict(X)
    np.save(os.path.join(RESULTS_DIR, 'cnm_pred_probs.npy'), y_probs)