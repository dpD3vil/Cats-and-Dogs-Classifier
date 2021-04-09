from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, InceptionV3, VGG19
import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model = tf.keras.models.load_model(
    r'model_2.h5')

test_image = image.load_img(
    r'TEST IMAGE PATH', target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image =  test_image / 255.0
result = np.argmax(model.predict(test_image))
if result == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
