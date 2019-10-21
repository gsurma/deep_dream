import numpy as np
from PIL import Image

def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    with open(filename, 'wb') as file:
        Image.fromarray(image).save(file, 'jpeg')

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min)

def resize_image(image, size=None, factor=None):
    if factor is not None:
        size = np.array(image.shape[0:2]) * factor
        size = size.astype(int)
    size = tuple(reversed(size)) # compensation for the numpy/PIL difference
    image = np.clip(image, 0.0, 255.0)
    image = Image.fromarray(image.astype(np.uint8))
    return np.float32(image.resize(size, Image.LANCZOS))