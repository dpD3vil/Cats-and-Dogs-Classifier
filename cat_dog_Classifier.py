import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop
import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

train_dir = r"PATH TO TRAINING DIR"
test_dir = r"PATH TO TEST DIR"
validation_dir = r"PATH TO VALIDATION DIR"
img_s = 200

Train_Datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
validation_Datagen = ImageDataGenerator(rescale=1./255)

train_generator = Train_Datagen.flow_from_directory(
    train_dir,
    target_size=(img_s, img_s),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = validation_Datagen.flow_from_directory(
    validation_dir,
    target_size=(img_s, img_s),
    batch_size=32,
    class_mode="categorical"
)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(img_s, img_s, 3), padding= 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation="relu", padding= 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(256, (3, 3), activation="relu", padding= 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(256, (3, 3), activation="relu", padding= 'same'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

#model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(0.001),
    metrics=['acc']
)

check_point = tf.keras.callbacks.ModelCheckpoint("model_2.h5", save_best_only= True)
earlystop = tf.keras.callbacks.EarlyStopping(patience=30)

history = model.fit(
    train_generator,
    steps_per_epoch=63,
    epochs=500,
    validation_data=validation_generator,
    validation_steps=31,
    callbacks = [check_point , earlystop]
)
