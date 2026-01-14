import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.utils.device import get_device
from src.models.xception import Xception
from src.datasets.food_dataset import FoodDataset


def train(
    csv_path="data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
    images_dir="data/raw/Food Images",
    labels_path="data/processed/labels.npy",
    num_classes=535,
    image_size=299,
    batch_size=16,
    epochs=10,
    lr=1e-4,
    checkpoint_dir="checkpoints",
):
    device = get_device()
    print("Using device:", device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    dataset = FoodDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        image_size=image_size,
        labels_path=labels_path,
    )

    print(
    "Dataset size:", len(dataset),
    "Labels shape:", dataset.labels.shape
)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.stack(labels)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,      # IMPORTANT for MPS stability
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # -------------------------
    # Model
    # -------------------------
    model = Xception(num_classes=num_classes)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # -------------------------
    # Sanity check (one batch)
    # -------------------------
    images, labels = next(iter(loader))
    print("Sanity check shapes:", images.shape, labels.shape)
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        print("Model output shape:", outputs.shape)

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{step}/{len(loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(
            checkpoint_dir, f"xception_epoch_{epoch+1}.pt"
        )
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)


if __name__ == "__main__":
    train(
        csv_path="data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
        images_dir="data/raw/Food Images",
        labels_path="data/processed/labels.npy",
        num_classes=535,   # MUST match labels.npy second dimension
        batch_size=16,
        epochs=10,
        lr=1e-4,
    )
