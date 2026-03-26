import os
import numpy as np

from glob import glob
from sklearn.model_selection import train_test_split

def load_data(path, split=0.1):
     X = sorted(glob(os.path.join(path, "images", "*.jpg")))
     Y = sorted(glob(os.path.join(path, "masks", "*.png")))
    
     train_x, test_x = train_test_split(X, test_size=split, random_state=42)
     train_y, test_y = train_test_split(Y, test_size=split, random_state=42)
     return (train_x, train_y), (test_x, test_y)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)