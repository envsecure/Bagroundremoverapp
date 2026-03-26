from sklearn.utils import shuffle
import os
import numpy as np
from glob import glob
import cv2
import tensorflow as tf
def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y
    
def read_image(path,H,W):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))  # Resize to (512, 512)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path,H,W):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))  # Resize to (512, 512)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y,H,W):
    def _parse(x, y):
        x = read_image(x,H,W)
        y = read_mask(y,H,W)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2, H=512, W=512):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf_parse(x, y, H, W))  # ✅ pass H, W here
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset