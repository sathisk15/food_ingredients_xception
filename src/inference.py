import json
import torch
import numpy as np
from PIL import Image
import re

from src.models.xception import Xception
from src.utils.device import get_device


def load_image(image_path, image_size):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    img = np.array(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img.unsqueeze(0)


def predict_ingredients(
    image_path,
    checkpoint_path,
    vocab_path,
    image_size=299,
    top_k=10,
):
    device = get_device()

    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    idx_to_ing = {idx: ing for ing, idx in vocab.items()}

    model = Xception(num_classes=len(vocab))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    image = load_image(image_path, image_size).to(device)

    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1]

    predictions = [(idx_to_ing[i], float(probs[i])) for i in top_indices]

    return predictions


if __name__ == "__main__":
    preds = predict_ingredients(
        image_path="data/raw/Food Images/sample.jpg",
        checkpoint_path="checkpoints/xception_epoch_1.pt",
        vocab_path="data/processed/ingredient_vocab.json",
        top_k=10,
    )

    for ing, score in preds:
        print(f"{ing}: {score:.3f}")
