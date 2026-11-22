"""Train headline classifiers and keep the best one.

Compares three algorithms on TF-IDF features:
    - Multinomial Naive Bayes
    - Logistic Regression
    - Linear SVM

Saves the highest-test-accuracy pipeline to models/model.pkl
and prints a full classification report.
"""

import sys
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_data

MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000, C=4.0),
    "linear_svm": LinearSVC(C=1.0),
}


def build_pipeline(model):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            ("clf", model),
        ]
    )


def main():
    train, test, target_names = load_data()
    print(f"Train: {len(train)} headlines | Test: {len(test)} headlines")
    print(f"Classes: {target_names}\n")

    best_name, best_acc, best_pipe = None, 0.0, None
    for name, model in MODELS.items():
        pipe = build_pipeline(model)
        t0 = time.time()
        pipe.fit(train["text"], train["target"])
        acc = accuracy_score(test["target"], pipe.predict(test["text"]))
        print(f"{name:20s} test accuracy = {acc:.4f}  ({time.time() - t0:.1f}s)")
        if acc > best_acc:
            best_name, best_acc, best_pipe = name, acc, pipe

    print(f"\nBest model: {best_name} ({best_acc:.4f})")
    print(
        classification_report(
            test["target"], best_pipe.predict(test["text"]), target_names=target_names
        )
    )
    joblib.dump(
        {"pipeline": best_pipe, "target_names": target_names}, "models/model.pkl"
    )
    print("Saved to models/model.pkl")


if __name__ == "__main__":
    sys.exit(main())
