from scripts.utils.ingredient_normalizer import clean_and_extract, filter_base_words, getSingularTokens, BAD_WORDS
import json
from collections import Counter
import pandas as pd
import json

CSV_PATH = "data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
OUT_PATH = "data/processed/ingredient_vocab.json"
TOP_K = 1000
MIN_FREQ = 2   # you can tune this (3–10)

BAD_SUFFIXES = (
    "ly",     # thinly, lightly, roughly
    "ed",     # toasted, chilled, beaten
)

def main():

    df = pd.read_csv(CSV_PATH)
    counter = Counter()
    all_tokens = []

    # Build frequency of clean ingredient words
    for ingredients in df["Cleaned_Ingredients"]:
        tokens = clean_and_extract(ingredients)
        base_tokens = filter_base_words(tokens)
        singular_tokens = getSingularTokens(base_tokens)
        all_tokens.extend(singular_tokens)
        counter.update(singular_tokens)

    filtered = []

    for word, count in counter.most_common():
        if word in BAD_WORDS:
            continue
        if word.endswith(BAD_SUFFIXES):
            continue
        if count < MIN_FREQ:
            continue
        filtered.append((word, count))

    most_common = filtered[:TOP_K]


    # most_common = counter.most_common(TOP_K)

    vocab = {
        ingredient: idx
        for idx, (ingredient, _) in enumerate(most_common)
    }

    print(f"Final vocabulary size: {len(vocab)}")

    with open(OUT_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print("Saved vocab to:", OUT_PATH)


if __name__ == "__main__":
    main()
