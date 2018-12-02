from os import listdir
from collections import defaultdict

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

def load_photos(directory, preprocess_func=None):
    images = list()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape(image.shape[0], image.shape[1], image.shape[2])
        # get image id
        images.append(image)

    images = np.array(images)

    if preprocess_func:
        images = preprocess_func(images.copy())
    return images

if __name__ == '__main__':
    images = load_photos('images/train',preprocess_input)
