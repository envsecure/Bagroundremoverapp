import os
from bg_remover.src.utils import load_data, torch_dataloader, shuffling
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]

CONFIG_DICT = ROOT_DIR / "configs" / "train.yaml"
DATA_DICT = ROOT_DIR / "configs" / "data.yaml"

with open(CONFIG_DICT) as f:
    cfg = yaml.safe_load(f)
with open(DATA_DICT) as f:
    data = yaml.safe_load(f)

""" Hyperparameters """
batch_size = cfg["batch_size"]
lr = cfg["lr"]
num_epochs = cfg["num_epochs"]


def train_data():
    """ Dataset """
    dataset_path = data["augmented_data_path"]
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "test")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_loader = torch_dataloader(train_x, train_y, batch=batch_size, shuffle_data=True)
    valid_loader = torch_dataloader(valid_x, valid_y, batch=batch_size, shuffle_data=False)
    return train_loader, valid_loader