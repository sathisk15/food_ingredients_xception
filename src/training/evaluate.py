import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from src.models.xception import Xception
from src.datasets.food_dataset import FoodDataset
from src.utils.device import get_device


def evaluate(
    csv_path,
    images_dir,
    labels_path,
    checkpoint_path,
    num_classes,
    image_size=299,
    batch_size=16,
    threshold=0.5,
):
    device = get_device()
    print("Using device:", device)

    dataset = FoodDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        image_size=image_size,
    )

    labels = np.load(labels_path)
    labels = labels[dataset.df.index.values]

    def collate_fn(batch):
        images, _ = zip(*batch)
        return torch.stack(images)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = Xception(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).cpu().numpy()
            targets = labels[
                i * batch_size : i * batch_size + images.size(0)
            ]

            all_preds.append(preds)
            all_targets.append(targets)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\nEvaluation results:")
    print(f"Micro Precision: {micro_p:.4f}")
    print(f"Micro Recall:    {micro_r:.4f}")
    print(f"Micro F1:        {micro_f1:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
