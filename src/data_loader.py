"""Shared data loading for AG News headline classification.

Label map (AG News convention):
    1 -> World, 2 -> Sports, 3 -> Business, 4 -> Sci/Tech
"""

import pandas as pd

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

TRAIN_PATH = "data/ag_news_sample.csv"
TEST_PATH = "data/ag_news_test.csv"


def load_data(train_path=TRAIN_PATH, test_path=TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for df in (train, test):
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
        df["target"] = df["label"].map(
            {v: k for k, v in enumerate(sorted(LABEL_MAP))}
        )
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    return train, test, target_names
