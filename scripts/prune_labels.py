import json
import numpy as np


LABELS_PATH = "data/processed/labels.npy"
VOCAB_PATH = "data/processed/ingredient_vocab.json"

OUT_LABELS_PATH = "data/processed/labels_pruned.npy"
OUT_VOCAB_PATH = "data/processed/ingredient_vocab_pruned.json"

MIN_COUNT = 50


def main():
    labels = np.load(LABELS_PATH)  # shape: [N, C]

    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    # index -> ingredient
    idx_to_ing = {idx: ing for ing, idx in vocab.items()}

    # count positives per label
    label_counts = labels.sum(axis=0)

    # keep labels with enough support
    keep_indices = [
        i for i, c in enumerate(label_counts) if c >= MIN_COUNT
    ]

    print(f"Original label count: {labels.shape[1]}")
    print(f"Kept label count: {len(keep_indices)}")

    # prune labels matrix
    labels_pruned = labels[:, keep_indices]

    # build new vocab
    vocab_pruned = {
        idx_to_ing[old_idx]: new_idx
        for new_idx, old_idx in enumerate(keep_indices)
    }

    np.save(OUT_LABELS_PATH, labels_pruned)

    with open(OUT_VOCAB_PATH, "w") as f:
        json.dump(vocab_pruned, f, indent=2)

    print("Saved:")
    print(" -", OUT_LABELS_PATH)
    print(" -", OUT_VOCAB_PATH)


if __name__ == "__main__":
    main()
