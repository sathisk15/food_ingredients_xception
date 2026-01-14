import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class FoodDataset(Dataset):
    """
    Returns:
        image_tensor: FloatTensor [3, H, W]
        label: FloatTensor [num_classes]
    """

    def __init__(
        self,
        csv_path,
        images_dir,
        image_size,
        labels_path,
        image_ids_path="data/processed/image_ids.txt",
    ):
        self.images_dir = images_dir
        self.image_size = image_size

        df = pd.read_csv(csv_path)
        labels = np.load(labels_path)

        # Load kept image IDs (AFTER pruning)
        with open(image_ids_path) as f:
            image_ids = [line.strip() for line in f]

        # Build image stem -> filename mapping
        image_map = {}
        for fname in os.listdir(images_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in {".jpg", ".jpeg", ".png"}:
                image_map[stem] = fname

        valid_rows = []
        valid_labels = []

        for idx, image_id in enumerate(image_ids):
            row = df[df["Image_Name"] == image_id]
            if len(row) == 0:
                continue
            if image_id not in image_map:
                continue

            valid_rows.append(row.iloc[0])
            valid_labels.append(labels[idx])

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.labels = torch.from_numpy(np.stack(valid_labels)).float()
        self.image_map = image_map

        assert len(self.df) == len(self.labels)

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_stem):
        image_path = os.path.join(self.images_dir, self.image_map[image_stem])
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        img = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        return img

    def __getitem__(self, idx):
        image_stem = self.df.iloc[idx]["Image_Name"]
        image = self._load_image(image_stem)
        label = self.labels[idx]
        return image, label
