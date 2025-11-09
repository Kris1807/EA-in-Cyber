import multiprocessing as mp
import sys
from pathlib import Path
import thrember

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download(file_type="Dot_Net", split="train"):
    print(f"Downloading {file_type} {split}...")
    thrember.download_dataset(str(DATA_DIR), file_type=file_type, split=split)
    print("✓ Done")

def load_data(subset="train"):
    print(f"Loading {subset}...")
    X, y = thrember.read_vectorized_features(str(DATA_DIR), subset=subset)
    print(f"  Loaded: {X.shape}")
    return X, y

def vectorize():
    print("Vectorizing...")
    thrember.create_vectorized_features(str(DATA_DIR))
    print("✓ Done")

if __name__ == "__main__":
    if sys.platform == "darwin":
        mp.set_start_method("spawn", force=True)

    print("Running EMBER2024 pipeline...")
    download(split="train")
    download(split="test")   # needed for vectorizer
    vectorize()
    load_data("train")
    load_data("test")
    print("\n✓ Pipeline complete!")
