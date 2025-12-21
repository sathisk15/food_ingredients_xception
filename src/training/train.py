import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.utils.device import get_device
from src.models.xception import Xception
from src.datasets.food_dataset import FoodDataset


def train(
    csv_path,
    images_dir,
    labels_path,
    num_classes,
    image_size=299,
    batch_size=16,
    epochs=10,
    lr=1e-4,
    checkpoint_dir="checkpoints",
):
    device = get_device()
    print("Using device:", device)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    dataset = FoodDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        image_size=image_size,
    )

    labels = np.load(labels_path)
    labels = labels[dataset.df.index.values]

    assert len(dataset) == labels.shape[0]

    def collate_fn(batch):
        images, _ = zip(*batch)
        images = torch.stack(images)
        return images

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # REQUIRED for MPS stability
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Model
    model = Xception(num_classes=num_classes)
    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for i, images in enumerate(loader):
            images = images.to(device)
            batch_labels = torch.from_numpy(
                labels[i * batch_size : i * batch_size + images.size(0)]
            ).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{i}/{len(loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} completed. Avg loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(checkpoint_dir, f"xception_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print("Saved checkpoint:", ckpt_path)
