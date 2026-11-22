from pathlib import Path
import joblib

try:
    from .preprocessing import clean_text
except ImportError:
    from preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "news_model.pkl"


def predict_news(news_text):
    """Predict whether a news article is fake or genuine."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train_model.py"
        )

    if not news_text or not news_text.strip():
        raise ValueError("Please enter some news text.")

    saved = joblib.load(MODEL_PATH)

    vectorizer = saved["vectorizer"]
    classifier = saved["classifier"]

    cleaned = clean_text(news_text)

    if not cleaned:
        raise ValueError("Please enter valid news content.")

    features = vectorizer.transform([cleaned])

    prediction = classifier.predict(features)[0]

    return prediction


if __name__ == "__main__":

    print("\n================================")
    print("     FAKE NEWS DETECTION SYSTEM")
    print("================================\n")

    title = input("Enter news headline: ").strip()
    article = input("Enter news article: ").strip()

    combined_news = title + " " + article

    try:
        result = predict_news(combined_news)

        print("\n========== PREDICTION ==========")

        if result == "fake":
            print("Result: FAKE NEWS")
        else:
            print("Result: GENUINE NEWS")

        print("================================")

    except (ValueError, FileNotFoundError) as error:
        print("Error:", error)