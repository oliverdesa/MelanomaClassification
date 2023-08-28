#import tensorflow as tf
#print(tf.__version__)
import os
import glob
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

# # convert the image pixels to a numpy array
# # encode the labels
# all_train_images = [image.img_to_array(img) for img in all_train_images]
# all_train_images = np.array(all_train_images).astype('float32')
# all_train_images /= 255.0

# encoder = LabelEncoder()
# all_train_labels = encoder.fit_transform(all_train_labels)
# all_train_labels = to_categorical(all_train_labels)

# all_test_images = [image.img_to_array(img) for img in all_test_images]
# all_test_images = np.array(all_test_images).astype('float32')
# all_test_images /= 255.0

# all_test_labels = encoder.fit_transform(all_test_labels)
# all_test_labels = to_categorical(all_test_labels)

# all_valid_images = [image.img_to_array(img) for img in all_valid_images]
# all_valid_images = np.array(all_valid_images).astype('float32')
# all_valid_images /= 255.0

# all_valid_labels = encoder.fit_transform(all_valid_labels)
# all_valid_labels = to_categorical(all_valid_labels)


# # reshape into a single sample with 3 channels
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# datagen.fit(all_train_images)


# # define the model
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# x = base_model.output
# x = Flatten()(x)
# x = Dense(512, activation='relu')(x)
# predictions = Dense(len(encoder.classes_), activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# # Freeze initial layers
# for layer in base_model.layers:
#     layer.trainable = False

# # compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # fit the model
# model.fit(datagen.flow(all_train_images, all_train_labels, batch_size=32),
#           epochs=5,
#           validation_data=(all_valid_images, all_valid_labels))

# # evaluate the model

