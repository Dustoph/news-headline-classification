"""Predict the category of a single news headline.

Usage:
    python src/predict.py "NASA unveils new Mars rover mission"
"""

import sys

import joblib


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    bundle = joblib.load("models/model.pkl")
    pipe, names = bundle["pipeline"], bundle["target_names"]
    headline = " ".join(sys.argv[1:])
    pred = pipe.predict([headline])[0]
    proba = (
        pipe.predict_proba([headline])[0]
        if hasattr(pipe.named_steps["clf"], "predict_proba")
        else None
    )
    print(f"Headline : {headline}")
    print(f"Category : {names[pred]}")
    if proba is not None:
        for name, p in sorted(zip(names, proba), key=lambda x: -x[1]):
            print(f"  {name:10s} {p:.3f}")


if __name__ == "__main__":
    main()
