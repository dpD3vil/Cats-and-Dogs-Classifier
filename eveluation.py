from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, InceptionV3, VGG19
import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#train_dir = r"C:\Users\dwark\Desktop\ML\Projects\Cats_Dogs Classifier\cats_and_dogs_small\train"
#test_dir = r"C:\Study\dks\ML\Projects\Cats_Dogs Classifier\cats_and_dogs_small\test"
#validation_dir = r"C:\Users\dwark\Desktop\ML\Projects\Cats_Dogs Classifier\cats_and_dogs_small\validation"

#Test_Datagen = ImageDataGenerator(rescale=1./255)
#Test_gen = Test_Datagen.flow_from_directory(
    #  test_dir,
    #target_size=(200, 200),
    #batch_size=20,
    #class_mode="binary",
    #shuffle=False)

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
