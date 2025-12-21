import json
import re
import numpy as np
import pandas as pd


CSV_PATH = "data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
VOCAB_PATH = "data/processed/ingredient_vocab.json"
OUT_LABELS = "data/processed/labels.npy"
OUT_IDS = "data/processed/image_ids.txt"


def clean_and_split(text: str):
    text = text.lower()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[0-9¼½¾⅓⅔⅛]+", "", text)
    parts = re.split(r",| and ", text)
    return [p.strip() for p in parts if p.strip()]


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
        tokens = clean_and_split(row["Ingredients"])

        for t in tokens:
            if t in vocab:
                labels[i, vocab[t]] = 1.0

    np.save(OUT_LABELS, labels)

    with open(OUT_IDS, "w") as f:
        for img_id in image_ids:
            f.write(f"{img_id}\n")

    print("Labels shape:", labels.shape)
    print("Saved:", OUT_LABELS)
    print("Saved:", OUT_IDS)


if __name__ == "__main__":
    main()
