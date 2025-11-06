import multiprocessing as mp
import sys

import thrember


def download(file_type="Dot_Net", split="train"):
    print(f"Downloading {file_type} {split}...")
    thrember.download_dataset("data", file_type=file_type, split=split)
    print("✓ Done")


def load_data(subset="train"):
    print(f"Loading {subset}...")
    X, y = thrember.read_vectorized_features("data", subset=subset)
    print(f"  Loaded: {X.shape}")
    return X, y


def vectorize():
    print("Vectorizing...")
    thrember.create_vectorized_features("data")
    print("✓ Done")


if __name__ == "__main__":
    if sys.platform == "darwin":
        mp.set_start_method("spawn", force=True)

    print("Running EMBER2024 pipeline...")
    download()
    vectorize()
    load_data("train")
    load_data("test")
    print("\n✓ Pipeline complete!")
