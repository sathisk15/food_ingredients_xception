import json
import re
from collections import Counter
import pandas as pd


CSV_PATH = "data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
OUT_PATH = "data/processed/ingredient_vocab.json"
TOP_K = 1000


def clean_and_split(text: str):
    text = text.lower()

    # remove parenthesized text
    text = re.sub(r"\([^)]*\)", "", text)

    # remove numbers and common fractions
    text = re.sub(r"[0-9¼½¾⅓⅔⅛]+", "", text)

    # split on commas and 'and'
    parts = re.split(r",| and ", text)

    return [p.strip() for p in parts if p.strip()]


def main():
    df = pd.read_csv(CSV_PATH)

    counter = Counter()

    for ingredients in df["Ingredients"]:
        tokens = clean_and_split(ingredients)
        counter.update(tokens)

    most_common = counter.most_common(TOP_K)

    vocab = {
        ingredient: idx
        for idx, (ingredient, _) in enumerate(most_common)
    }

    print(f"Final vocabulary size: {len(vocab)}")

    with open(OUT_PATH, "w") as f:
        json.dump(vocab, f, indent=2)


if __name__ == "__main__":
    main()
