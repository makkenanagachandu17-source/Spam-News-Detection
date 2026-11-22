from src.predict import predict_news

def main():
    print("=" * 45)
    print("       SPAM / FAKE NEWS DETECTION")
    print("=" * 45)
    print("Paste a news article or sentence. Type 'exit' to quit.")
    while True:
        article = input("\\nNews text: ").strip()
        if article.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        try:
            print("Prediction:", predict_news(article).upper())
        except Exception as exc:
            print("Error:", exc)

if __name__ == "__main__":
    main()
