import sys

sys.path.append('.')
from keras.models import load_model
from cnn_image_utils import load_photos

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler('tries.log'))

if __name__ == '__main__':

    model = load_model('tuned/MobileNetV2_tuned.h5')

    images = load_photos("images/real")
    logger.info("Loaded {} real images".format(len(images)))

    predicted = model.predict(images)
    print(predicted)
    logging.info(str(predicted))

    images = load_photos("images/fakes")
    logger.info("Loaded {} fakes images".format(len(images)))

    predicted = model.predict(images)
    print(predicted)
    logging.info(str(predicted))
