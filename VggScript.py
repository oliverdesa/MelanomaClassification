import tensorflow as tf
from keras.applications.vgg16 import VGG16
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
import my_functions as f
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions 

# load the model
model = VGG16(weights='imagenet')

# loop through each subdirectory collecting images and respective labels
train_images, train_labels = f.load_images('../data/skin-lesions/train/')
test_images, test_labels = f.load_images('../data/skin-lesions/test/')
valid_images, valid_labels = f.load_images('../data/skin-lesions/valid/')

# encode the labels
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_labels)
train_labels = to_categorical(train_labels)

test_labels = encoder.fit_transform(test_labels)
test_labels = to_categorical(test_labels)

valid_labels = encoder.fit_transform(valid_labels)
valid_labels = to_categorical(valid_labels)


# reshape into a single sample with 3 channels
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_images)


# define the model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(len(encoder.classes_), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze initial layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
          epochs=5,
          validation_data=(valid_images, valid_labels))

# evaluate the model

