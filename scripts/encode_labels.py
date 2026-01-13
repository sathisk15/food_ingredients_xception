from scripts.utils.ingredient_normalizer import clean_and_extract, filter_base_words, getSingularTokens
import json
import re
import numpy as np
import pandas as pd


CSV_PATH = "data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
VOCAB_PATH = "data/processed/ingredient_vocab.json"
OUT_LABELS = "data/processed/labels.npy"
OUT_IDS = "data/processed/image_ids.txt"

def main():
    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    df = pd.read_csv(CSV_PATH)

    num_samples = len(df)
    num_labels = len(vocab)

    labels = np.zeros((num_samples, num_labels), dtype=np.float32)

    image_ids = []

    for i, row in df.iterrows():
        image_ids.append(row["Image_Name"])
        base_tokens = clean_and_extract(row["Cleaned_Ingredients"])
        base_tokens = filter_base_words(base_tokens)
        tokens = getSingularTokens(base_tokens)
        for t in tokens:
            if t in vocab:
                labels[i, vocab[t]] = 1.0
    
    # --- DROP ZERO-LABEL IMAGES ---
    label_sums = labels.sum(axis=1)
    valid_mask = label_sums > 0

    labels = labels[valid_mask]
    image_ids = [img for img, keep in zip(image_ids, valid_mask) if keep]

    np.save(OUT_LABELS, labels)

    with open(OUT_IDS, "w") as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")

    print("Labels shape:", labels.shape)
    print("Saved:", OUT_LABELS)
    print("Saved:", OUT_IDS)


if __name__ == "__main__":
    main()
