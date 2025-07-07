import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize
import joblib

from utils import load_model

model = load_model()

file_path = input("Enter the destination of the input file: ")

img = mpimg.imread(file_path)

if img.ndim == 3:  # Check if it's RGB or RGBA
    # Drop alpha channel if present
    if img.shape[2] == 4:
        img = img[:, :, :3]

    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # standard weights
else:
    gray = img  # already grayscale

gray_resized = resize(gray, (28, 28), anti_aliasing=True)
# gray = gray[gray.shape[0] // 28 - 1::gray.shape[0] // 28, gray.shape[1] // 28 - 1::gray.shape[1] // 28]

pred = model.predict(gray_resized.reshape(1, -1))

print(pred[0])