import os
import numpy as np
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y


def read_image(path, H, W):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    # HWC → CHW (PyTorch convention)
    x = np.transpose(x, (2, 0, 1))
    return x


def read_mask(path, H, W):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    # (H, W) → (1, H, W)
    x = np.expand_dims(x, axis=0)
    return x


class SegmentationDataset(Dataset):
    """PyTorch Dataset replacement for the old tf.data.Dataset pipeline."""

    def __init__(self, image_paths, mask_paths, H=512, W=512):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.H = H
        self.W = W

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = read_image(self.image_paths[idx], self.H, self.W)
        mask = read_mask(self.mask_paths[idx], self.H, self.W)
        return torch.from_numpy(image), torch.from_numpy(mask)


def torch_dataloader(X, Y, batch=2, H=512, W=512, shuffle_data=False, num_workers=4):
    """Drop-in replacement for the old tf_dataset function."""
    dataset = SegmentationDataset(X, Y, H=H, W=W)
    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle_data,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


# Keep old name as alias for backward compatibility
tf_dataset = torch_dataloader