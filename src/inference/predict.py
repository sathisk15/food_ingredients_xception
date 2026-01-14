import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.models.xception import Xception
from src.utils.device import get_device


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def load_image(image_path, image_size=299):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((image_size, image_size))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    return img, img_tensor.unsqueeze(0)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict_ingredients(
    image_path,
    checkpoint_path,
    vocab_path,
    image_size=299,
    top_k=10,
    min_prob=0.05,
):
    device = get_device()
    print("Using device:", device)

    # Load vocab
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    idx_to_ing = {int(v): k for k, v in vocab.items()}

    # Load model
    model = Xception(num_classes=len(vocab))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image
    original_img, image_tensor = load_image(image_path, image_size)
    image_tensor = image_tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    top_indices = probs.argsort()[-top_k:][::-1]

    predictions = [
        (idx_to_ing[i], float(probs[i]))
        for i in top_indices
        if probs[i] >= min_prob
    ]

    return original_img, predictions


# --------------------------------------------------
# Display image + predictions
# --------------------------------------------------
def show_results(image, predictions):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")

    if predictions:
        title = ", ".join(
            [f"{ing} ({score:.2f})" for ing, score in predictions[:5]]
        )
    else:
        title = "No confident ingredients detected"

    plt.title(title, fontsize=10)
    plt.show()


# --------------------------------------------------
# Run directly
# --------------------------------------------------
if __name__ == "__main__":
    image_path = "data/raw/Food Images/zucchini-salad-with-ajo-blanco-dressing-spiced-nuts-56389847.jpg"
    checkpoint_path = "checkpoints/xception_epoch_9.pt"
    vocab_path = "data/processed/ingredient_vocab.json"

    image, preds = predict_ingredients(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        top_k=10,
        min_prob=0.05,
    )

    print("\nPredicted ingredients:")
    for ing, score in preds:
        print(f"• {ing:<15} {score:.3f}")

    show_results(image, preds)
