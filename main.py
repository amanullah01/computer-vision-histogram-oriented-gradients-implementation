from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np


# Now we can import human faces
human_faces = fetch_lfw_people()
positive_images = human_faces.images[:10000]
print(positive_images.shape)
plt.imshow(positive_images[1], cmap='gray')
plt.show()
