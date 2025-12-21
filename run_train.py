from src.training.train import train


if __name__ == "__main__":
    train(
        csv_path="data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv",
        images_dir="data/raw/Food Images",
        labels_path="data/processed/labels_pruned.npy",
        num_classes=400,
        image_size=299,
        batch_size=16,
        epochs=10,
        lr=1e-4,
        checkpoint_dir="checkpoints_pruned",
    )
