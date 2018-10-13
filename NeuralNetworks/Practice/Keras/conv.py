# https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import numpy
import theano

import numpy as np
from matplotlib import pyplot as plt

# Simple linear stack of neural network layer
# Perfect for the type of FF CNN
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPool2D
from keras.utils import np_utils

from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])