from pathlib import Path
import sys
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from preprocessing import load_dataset


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "dataset" / "WELFake_Dataset.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def main():

    # Load and preprocess dataset
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH

    print("Loading dataset...")
    df = load_dataset(data_path)

    print(f"Total records: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain exactly two classes.")

    # Separate features and labels
    X = df["news"]
    y = df["label"]

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # TF-IDF feature extraction
    print("\nExtracting TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=2,
        ngram_range=(1, 1)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Passive Aggressive Classifier
    print("\nTraining Passive Aggressive Classifier...")

    classifier = PassiveAggressiveClassifier(
        max_iter=50,
        random_state=42
    )

    classifier.fit(X_train_tfidf, y_train)

    # Predictions
    print("\nEvaluating model...")

    predictions = classifier.predict(X_test_tfidf)

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    }

    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "classifier": classifier
        },
        MODEL_DIR / "news_model.pkl"
    )

    # Save evaluation report
    with open(REPORT_DIR / "metrics.txt", "w", encoding="utf-8") as file:

        file.write("Spam / Fake News Detection - Evaluation Report\n\n")

        for name, value in metrics.items():
            file.write(f"{name}: {value:.4f}\n")

        file.write("\nClassification Report:\n")
        file.write(
            classification_report(y_test, predictions, zero_division=0)
        )

    # Generate confusion matrix
    labels = ["fake", "true"]

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions, labels=labels),
        display_labels=labels
    ).plot()

    plt.title("Fake News Detection - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # Display results
    print("\n========== TRAINING RESULTS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f} ({value * 100:.2f}%)")

    print("\nTraining completed successfully!")

    print(f"\nModel saved at: {MODEL_DIR / 'news_model.pkl'}")
    print(f"Metrics saved at: {REPORT_DIR / 'metrics.txt'}")
    print(f"Confusion matrix saved at: {REPORT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()