# News Headline Classification Using Machine Learning

Classify news headlines into **World, Sports, Business,** or **Sci/Tech**
using TF-IDF features and classical ML models. Built on the
[AG News](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) dataset.

## Results

| Model | Test accuracy (7,600 headlines) |
|---|---|
| Multinomial Naive Bayes | 0.8942 |
| **Logistic Regression** | **0.8954** |
| Linear SVM | 0.8949 |

Best model (Logistic Regression) per-class F1: Sports 0.95, World 0.90,
Sci/Tech 0.87, Business 0.86. Macro F1 ≈ 0.90.

## Project structure

```
news-headline-classification/
├── data/
│   ├── ag_news_sample.csv   # 12,000 headlines, stratified (3,000/class)
│   └── ag_news_test.csv     # 7,600 held-out headlines for evaluation
├── src/
│   ├── data_loader.py       # CSV loading + label mapping
│   ├── train.py             # trains NB / LogReg / LinearSVC, saves best
│   └── predict.py           # single-headline CLI prediction
├── notebooks/
│   └── kaggle_notebook.ipynb  # Kaggle-ready copy of this pipeline
└── models/                  # created by train.py (model.pkl)
```

## Quickstart

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py "NASA unveils new Mars rover mission"
```

## How it works

1. Headline + description are concatenated into one text field.
2. Text is vectorized with TF-IDF (word unigrams + bigrams, 20k features,
   English stop words removed, sublinear term frequencies).
3. Three classifiers are trained on 12,000 headlines and compared on the
   7,600-headline test set; the best pipeline is saved to `models/model.pkl`.

## Kaggle notebook

`notebooks/kaggle_notebook.ipynb` is self-contained (downloads the public
AG News CSVs itself, needs no API keys). Upload it via Kaggle → New Notebook → Upload,
or paste the notebook URL via File → Import from URL.

## Dataset credit

AG News corpus (Gulli). The CSVs here are a stratified subset of the
public 120k/7.6k train/test splits (mirrored for this project from the
CharCnn_Keras AG News CSV release).
