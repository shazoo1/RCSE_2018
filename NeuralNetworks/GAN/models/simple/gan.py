import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras import Sequential, Model
from keras.layers import Input, InputLayer, Dense,LeakyReLU, Activation, Dropout, Flatten,Reshape, BatchNormalization
from keras.optimizers import Adam
from keras.datasets import mnist

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class GAN:
    def __init__(self):
        self.g_input_shape = 100
        self.d_input_shape = (28,28)

        self.g_output_shape = (28,28)
        self.d_output_shape = 1
        self.img_shape = (28,28, 1)

        self.epochs = 30000
        self.batch_size = 128

        self.optimizer = Adam(0.0002, 0.5)

        self.gen = self.get_generator()
        self.dis = self.get_discriminator()
        self.gan = self.get_combined()

        self.g_losses = []
        self.d_losses = []
        self.accuracy = []



    def get_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.g_input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.g_input_shape,))
        img = model(noise)

        return Model(noise, img)

    def get_discriminator(self):
        model = Sequential()


        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def get_combined(self):

        # First comply dis
        self.dis.compile(loss='binary_crossentropy', optimizer=self.optimizer,
                         metrics=['accuracy'])


        # Prepare generator
        gan_input = Input(shape=[self.g_input_shape,])
        img = self.gen(gan_input)


        # Prepare discriminator
        self.dis.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.dis(img)

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        combined = Model(gan_input, validity)
        combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        combined.summary()

        return combined

    def train(self):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial labels
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        print('Epochs:', self.epochs)
        print('Batch size:', self.batch_size)
        print("Batches per epoch:", len(X_train) / self.batch_size)

        for epoch in range(self.epochs):

            #print("-" * 15, 'Epoch %d' % epoch, "-" * 15)

            #for _ in tqdm(range(len(X_train) // self.batch_size)):

            # ---------------------
            #  Preparation stage
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]
            # Generate a batch of fake images
            noise = np.random.normal(0, 1, (self.batch_size, self.g_input_shape))
            gen_imgs = self.gen.predict(noise)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # First real data, then fake data
            self.dis.trainable = True
            d_loss_real = self.dis.train_on_batch(imgs, valid)
            d_loss_fake = self.dis.train_on_batch(gen_imgs, fake)
            d_loss = d_loss_fake+d_loss_real
            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)
            self.dis.trainable = False
            noise = np.random.normal(0, 1, (self.batch_size, self.g_input_shape))
            g_loss = self.gan.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))


            if epoch % 100 == 0:
                # Store loss of most recent batch from this epoch
                self.g_losses.append(g_loss)
                self.d_losses.append(d_loss)
                self.accuracy.append(100*d_loss[1])
                #self.plot_generated_images(epoch)
                #self.save_models(epoch)
                self.sample_images(epoch)
                self.save_graphics(epoch)
                #Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    def save_models(self, epoch):
        self.gen.save('models/gan_generator_epoch_%d.h5' %epoch)
        self.dis.save('models/gan_discriminator_epoch_%d.h5' % epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.g_input_shape))
        gen_imgs = self.gen.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_graphics(self,epoch):

        plt.figure(1)

        # Plot accurancy
        plt.subplot(221)
        plt.title("Accurancy")
        plt.plot(self.accuracy)

        # Plot G_loss
        plt.subplot(222)
        plt.title("G_loss")
        plt.plot(self.g_losses)

        # Plot D_loss
        plt.subplot(223)
        plt.title("D_loss")
        plt.plot(self.d_losses)

        plt.tight_layout()
        plt.savefig('graphs/graphs_%d.png' % epoch)

    def plot_generated_images(self, epoch, examples=100, dim=(10,10), figsize=(10,10)):

        noise = np.random.normal(0,1, size=[examples, self.g_input_shape])

        generated_images = self.gen.predict(noise)
        generated_images = generated_images.reshape(examples, 28, 28)

        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i],
                       interpolation='nearest',
                       cmap='gray_r',)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('images/gan_generated_image_epoch_%d.png' %epoch)


if __name__ == '__main__':
    gan = GAN()
    gan.train()