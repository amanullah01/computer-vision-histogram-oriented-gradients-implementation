from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from skimage.io import imread
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np

# Now we can import human faces
human_faces = fetch_lfw_people()
positive_images = human_faces.images[:10000]
print(positive_images.shape)

# fetch datasets non faces
non_face_topics = ['moon', 'text', 'coins']
negative_images_samples = [(getattr(data, name)()) for name in non_face_topics]


# plt.imshow(positive_images[1], cmap='gray')
# plt.show()
# for image in negative_images_samples:
#     plt.imshow(image)
#     plt.show()


# we will use now patch extractor to generate several combinations of images
def generate_random_sample(image, num_of_generated_images=100, patch_size=positive_images[0].shape):
    extractor = PatchExtractor(patch_size=patch_size, max_patches=num_of_generated_images, random_state=42)
    patches = extractor.transform(image[np.newaxis])
    return patches


# now we will generate 3000 images
negative_images = np.vstack([generate_random_sample(im, 1000) for im in negative_images_samples])
print(negative_images.shape)

"""
fig, ax = plt.subplots(10, 10)
for i, axis in enumerate(ax.flat):
    axis.imshow(negative_images[2000+i], cmap='gray')
    axis.axis('off')

plt.show()
"""

# we construct the training set with the output variable
# we have to construct the HOG features
# time-consuming
X_train = np.array([feature.hog(image) for image in chain(positive_images, negative_images)])
# label 0 - 1 // 0: non-face, 1: positive face

y_train = np.zeros(X_train.shape[0])
y_train[:positive_images.shape[0]] = 1

# we construct the SVM
svm = LinearSVC()
svm.fit(X_train, y_train)

# read the test images
test_image = imread(fname="images/girl_face.png")
test_image = transform.resize(test_image, positive_images[0].shape)

plt.imshow(test_image, cmap='gray')
plt.show()

test_image_hog = np.array([feature.hog(test_image, channel_axis=-1)])
prediction = svm.predict(test_image_hog)
print(prediction)
print("Prediction made by SVM: %f" % prediction)
