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
        ingredients_raw: str
    """

    def __init__(self, csv_path, images_dir, image_size):
        self.images_dir = images_dir
        self.image_size = image_size

        df = pd.read_csv(csv_path)

        required_cols = {"Image_Name", "Ingredients"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        # Build image stem -> filename mapping
        image_map = {}
        for fname in os.listdir(images_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in {".jpg", ".jpeg", ".png"}:
                image_map[stem] = fname

        # Filter rows with valid images only
        valid_rows = []
        for _, row in df.iterrows():
            stem = row["Image_Name"]
            if isinstance(stem, str) and stem in image_map:
                valid_rows.append(row)

        if len(valid_rows) == 0:
            raise RuntimeError("No valid image entries found after filtering")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.image_map = image_map

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_stem):
        image_path = os.path.join(self.images_dir, self.image_map[image_stem])

        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.image_size, self.image_size))

        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_stem = row["Image_Name"]
        ingredients_raw = row["Ingredients"]

        image = self._load_image(image_stem)

        return image, ingredients_raw
  