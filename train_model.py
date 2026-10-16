
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Project folder
BASE_DIR = Path(__file__).resolve().parent

FAKE_FILE = BASE_DIR / "Fake.csv"
REAL_FILE = BASE_DIR / "True.csv"
MODEL_FILE = BASE_DIR / "fake_news_model.pkl"

SAMPLES_PER_CLASS = 2500

print("Step 1: Checking dataset files...")

if not FAKE_FILE.exists() or not REAL_FILE.exists():
    raise FileNotFoundError(
        "Fake.csv and True.csv must be inside: "
        + str(BASE_DIR)
    )


def load_news(file_path, label):
    print(f"Reading {file_path.name}...")

    data = pd.read_csv(
        file_path,
        encoding="utf-8",
        nrows=15000,
        on_bad_lines="skip"
    )

    # Normalize column names
    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    if "title" not in data.columns or "text" not in data.columns:
        raise ValueError(
            f"Expected title and text columns in {file_path.name}. "
            f"Found: {data.columns.tolist()}"
        )

    data = data[["title", "text"]].fillna("")

    data["content"] = (
        data["title"].astype(str)
        + " "
        + data["text"].astype(str)
    ).str.strip()

    data = data[data["content"].str.len() > 20]
    data = data.drop_duplicates(subset=["content"])

    # Label: 0 = Fake, 1 = Real
    data["label"] = label

    # Sample articles randomly
    count = min(SAMPLES_PER_CLASS, len(data))
    data = data.sample(n=count, random_state=42)

    print(f"Loaded {len(data)} articles from {file_path.name}")

    return data[["content", "label"]]


print("Step 2: Loading fake and real news...")

fake_news = load_news(FAKE_FILE, 0)
real_news = load_news(REAL_FILE, 1)

# Combine both classes
df = pd.concat(
    [fake_news, real_news],
    ignore_index=True
)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset summary:")
print(df["label"].value_counts())

if df["label"].nunique() != 2:
    raise ValueError("Dataset must contain both fake and real articles.")

print("\nStep 3: Splitting training and testing data...")

X_train, X_test, y_train, y_test = train_test_split(
    df["content"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("Step 4: Creating machine learning model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

print("Step 5: Training model. Please wait...")

model.fit(X_train, y_train)

print("\nStep 6: Evaluating model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions))

print("Step 7: Saving trained model...")

joblib.dump(model, MODEL_FILE)

print(f"\nModel saved successfully: {MODEL_FILE}")
print("Training completed successfully!")
