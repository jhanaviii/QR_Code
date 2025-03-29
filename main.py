from traditional_ml import train_and_save_ml_model
from deep_learning import train_and_save_cnn_model
from evaluation import evaluate_ml_model, evaluate_cnn_model


def main():
    print("=== Training ML Model ===")
    train_and_save_ml_model()

    print("\n=== Training CNN Model ===")
    train_and_save_cnn_model()

    print("\n=== Evaluating Models ===")
    evaluate_ml_model()
    evaluate_cnn_model()


if __name__ == "__main__":
    main()