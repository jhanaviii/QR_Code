import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_utils import load_images
from utils.feature_utils import extract_all_features
from config import ML_MODEL_PATH


def train_and_save_ml_model():
    # Load data
    originals, counterfeits = load_images()

    # Verify data loading
    if len(originals) == 0 or len(counterfeits) == 0:
        raise ValueError("No images loaded. Check your data directory structure and files")

    # Extract features
    print("Extracting features from originals...")
    X_orig = extract_all_features(originals)
    print("Extracting features from counterfeits...")
    X_fake = extract_all_features(counterfeits)

    # Create labels
    X = np.vstack((X_orig, X_fake))
    y = np.array([0] * len(X_orig) + [1] * len(X_fake))
    print(f"Final dataset shape: {X.shape}, labels: {y.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Rest of your training code...

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"ML Model Accuracy: {accuracy:.4f}")

    # Save model
    joblib.dump(model, ML_MODEL_PATH)
    print(f"Model saved to {ML_MODEL_PATH}")