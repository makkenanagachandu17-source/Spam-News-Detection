import re
import pandas as pd


def clean_text(text):
    """Clean and normalize news text."""

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset(path):
    """Load and preprocess the WELFake news dataset."""

    df = pd.read_csv(path)

    required_columns = ["title", "text", "label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Dataset must contain a '{column}' column.")

    # Remove records with missing labels.
    df = df.dropna(subset=["label"]).copy()

    # Replace missing titles and article text.
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title and article text.
    df["news"] = df["title"].astype(str) + " " + df["text"].astype(str)

    # Clean combined news content.
    df["news"] = df["news"].apply(clean_text)

    # Remove empty news records.
    df = df[df["news"].str.len() > 0].copy()

    # Convert numeric labels into readable categories.
    df["label"] = df["label"].astype(int).map({
        0: "true",
        1: "fake"
    })

    df = df.dropna(subset=["label"])

    return df[["news", "label"]].reset_index(drop=True)