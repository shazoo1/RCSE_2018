from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg16 import VGG16

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Activation
import matplotlib.pyplot as plt
from os import listdir
import numpy as np

from keras import backend as K

# to evaluate time of prediction
import time

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('metrics.log'))


train_dir = r"images\train"
validation_dir = r"images\validation"
image_size = 224

def get_model(pretrained_model):
    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(pretrained_model)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.7))
    model.add(layers.Dense(1))
    model.add(Activation('sigmoid'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model


def load_photos(directory, preprocess_func=None):
    images = list()
    for name in listdir(directory):
        # load an image from file
        filename = directory + '/' + name
        image = load_img(filename, target_size=(image_size, image_size))
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

def predict_for_time(model):
    fake_data = 'images/fakes'
    real_data = 'images/real'

    fake_images = load_photos(fake_data)
    real_images = load_photos(real_data)

    # First evaluate on fake data

    accuracy = []
    loss = []
    avg_time =[]

    for image in fake_images:
        image = np.expand_dims(image, 0)
        start = time.time()
        y_false = np.zeros((1,))
        fake_score = model.evaluate(image, y_false)
        end = time.time() - start

        accuracy.append(fake_score[0])
        loss.append(fake_score[1])
        avg_time.append(end)

    fake_score = (np.mean(accuracy), np.mean(loss), np.mean(avg_time))

    # Then evaluate on real data

    accuracy = []
    loss = []
    avg_time = []

    for image in real_images:
        image = np.expand_dims(image,0)
        start = time.time()
        y_real = np.ones((1,))
        real_score = model.evaluate(image, y_real)
        end = time.time() - start

        accuracy.append(real_score[0])
        loss.append(real_score[1])
        avg_time.append(end)

    real_score = (np.mean(accuracy), np.mean(loss), np.mean(avg_time))

    return fake_score, real_score

def describe_metric(model_name, metrics):
    logger.info("%s model loss: %s" % (model_name,metrics[0]))
    logger.info("%s accuracy: %s" % (model_name,metrics[1]))
    logger.info("%s elapsed total elapsed time: %s" % (model_name,metrics[2]))

if __name__ == '__main__':


    #prepared_models = [VGG16, MobileNetV2 ,InceptionResNetV2, MobileNet, ResNet50 ]
    #name_models = ["VGG16", "MobileNetV2" ,"InceptionResNetV2", "MobileNet", "ResNet50" ]
    prepared_models = [ MobileNetV2, ]
    name_models = [ "MobileNetV2",]
    index = 0
    for _name, _model in zip(name_models, prepared_models):

        _pretrained = _model(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

        model = get_model(_pretrained)
        # Freeze the layers except the last 4 layers
        for layer in _pretrained.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        #for layer in _pretrained.layers:
        #    print(layer, layer.trainable)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            )

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        # Change the batchsize according to your system RAM
        train_batchsize = 5
        val_batchsize = 5



        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(image_size, image_size),
            batch_size=train_batchsize,
            class_mode='binary')

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(image_size, image_size),
            batch_size=val_batchsize,
            class_mode='binary',
            shuffle=False)

        # Compile the model
        model.compile(loss='binary_crossentropy',
                      #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      optimizer='adam',
                      metrics=['accuracy'])
        # Train the model
        history = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples / train_generator.batch_size,
            epochs=30,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples / validation_generator.batch_size,
            verbose=0)

        logger.debug(history)

        # Save the model
        model.save('tuned/{}_tuned.h5'.format(_name))

        # Get information on trained model, it's speed, accuracy on real dataset
        logger.info("<- Describing metrics for {}, training index is {} ->".format(_name,index))
        fake_metric, real_metric = predict_for_time(model)
        logger.info('Metrics on fake images')
        describe_metric(_name,fake_metric)
        logger.info('Metrics on real images with noise')
        describe_metric(_name,real_metric)
        logger.info('<- ->')


        if history:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs = range(len(acc))
            plt.figure()

            plt.subplot(2,1,1)

            plt.plot(epochs, acc, 'b', label='Training acc')
            plt.plot(epochs, val_acc, 'r', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.legend()

            plt.subplot(2, 1, 2)

            plt.plot(epochs, loss, 'b', label='Training loss')
            plt.plot(epochs, val_loss, 'r', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend()

            #plt.show()

            plt.savefig('metrics/{}.jpg'.format(_name))
            plt.close()

        # free memory for GPU as it is still in memory hence training might be failed
        K.clear_session()
        index+=1