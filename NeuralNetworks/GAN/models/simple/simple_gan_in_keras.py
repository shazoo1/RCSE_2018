
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input,Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D,Conv1D , Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard

from keras.datasets import fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#http://www.rricard.me/machine/learning/generative/adversarial/networks/keras/tensorflow/2017/04/05/gans-part2.html

def generator_model(G_in, dense_dim=200, out_dim=50, lr=1e-3):

    model = Sequential()

    model.add(Dense(dense_dim))
    model.add(Activation('relu'))
    model.add()
    G_out = Dense(out_dim, activation='sigmoid')(x)

    G = Model(G_in, G_out)

    opt = SGD(lr=lr)

    G.compile(loss='binary_crossentropy', optimizer=opt)

    return G, G_out

G_in = Input(shape=[28,28])
G, G_out = generator_model(G_in, out_dim=28*28)
G.summary()


def discriminator_model(D_in, lr=1e-3, drate=.25,
                        n_channels=50,
                        conv_sz=5,
                        leak=.2):
    x = Reshape((-1,1))(D_in)
    x = Conv1D(n_channels,
               conv_sz,
               activation='relu')(x)
    x = Dropout(drate)(x)
    x = Flatten()(x)
    x = Dense(n_channels)(x)

    D_out = Dense(2, activation='sigmoid')(x)

    D = Model(D_in, D_out)

    dopt = Adam(lr=lr)

    D.compile(loss='binary_crossentropy', optimizer=dopt)

    return D, D_out

D_in = Input(shape=[28,28])
D, D_out = discriminator_model(D_in)

D.summary()


def set_trainability(model, trainable=False):
    model.trainable = trainable

    for layer in model.layers:
        layer.trainable = trainable

def make_gan(GAN_in, G, D):
    set_trainability(D, False)

    x = G(GAN_in)
    GAN_out = D(x)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss='binary_crossentropy', optimizer=G.optimizer)
    return GAN, GAN_out


GAN_in = Input([28,28])
GAN, GAN_out = make_gan(GAN_in, G, D)
GAN.summary()



def sample_data_and_gen(G, noise_dim=10, n_samples=10000):
    XT = train_images

    XN_noise = np.random.uniform(0,1,size=[n_samples,noise_dim])
    XN = G.predict(XN_noise)

    X = np.concatenate((XT,XN))
    y = np.zeros((2*n_samples,2))
    y[:n_samples,1] = 1
    y[n_samples:,0] = 1

    return X,y

def pretrain(G, D, noise_dim=10, n_samples=10000, batch_size=32):
    X, y = sample_data_and_gen(G, n_samples, noise_dim)
    set_trainability(D, True)
    D.fit(X,y, epochs=1 ,batch_size=batch_size)

pretrain(G,D)