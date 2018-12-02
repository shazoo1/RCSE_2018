from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.vgg16 import VGG16


from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt


train_dir = r"C:\Users\user\PycharmProjects\RCSE2018\NeuralNetworks\Practice\Practise4\images\train"
validation_dir = r"C:\Users\user\PycharmProjects\RCSE2018\NeuralNetworks\Practice\Practise4\images\validation"
image_size = 32

def get_model(pretrained_model):
    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(pretrained_model)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Show a summary of the model. Check the number of trainable parameters
    #model.summary()

    return model




vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

model = get_model(vgg_conv)
# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Change the batchsize according to your system RAM
train_batchsize = 1
val_batchsize = 1



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
              optimizer=optimizers.RMSprop(lr=1e-4),
              #optimizer='adam',
              metrics=['acc'])
# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples / train_generator.batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples / validation_generator.batch_size,
    verbose=0)

# Save the model
model.save('small_last4.h5')

print(history.history)
print(dir(history.history))

if history:
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    #plt.show()

    plt.savefig('accur')