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

# load the model
model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions  

# load an image from file
#imge_path = 'C:/Users/odesa/Desktop/Melanoma/skin_lesion/train/*.*
# all_images = []
# for img_path in glob.glob(imge_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     all_images.append(img)

#print(glob.glob('skin-lesions\\train\\melanoma\\*.jpg'))


def load_images(image_path, label):
    all_images = []
    labels = []
    for file in glob.glob(image_path):
        img = image.load_img(file, target_size=(224, 224))
        all_images.append(img)
        labels.append(label)
    return all_images, labels

train_melanoma_imgs, train_melanoma_labels = load_images('..\\data\\skin-lesions\\train\\melanoma\\*.jpg', 'melanoma')
train_nevus_imgs, train_nevus_labels = load_images('..\\data\\skin-lesions\\train\\nevus\\*.jpg', 'nevus')
train_seborrheic_keratosis_imgs, train_seborrheic_keratosis_labels = load_images('..\\data\\skin-lesions\\train\\seborrheic_keratosis\\*.jpg', 'seborrheic_keratosis')
    
all_train_images = train_melanoma_imgs + train_nevus_imgs + train_seborrheic_keratosis_imgs
all_train_labels = train_melanoma_labels + train_nevus_labels + train_seborrheic_keratosis_labels

test_melanoma, test_melanoma_labels = load_images('..\\data\\skin-lesions\\test\\melanoma\\*.jpg', 'melanoma')
test_nevus, test_nevus_labels = load_images('..\\data\\skin-lesions\\test\\nevus\\*.jpg', 'nevus')
test_seborrheic_keratosis, test_seborrheic_keratosis_labels = load_images('..\\data\\skin-lesions\\test\\seborrheic_keratosis\\*.jpg', 'seborrheic_keratosis')

all_test_images = test_melanoma + test_nevus + test_seborrheic_keratosis
all_test_labels = test_melanoma_labels + test_nevus_labels + test_seborrheic_keratosis_labels

valid_melanoma, valid_melanoma_labels = load_images('..\\data\\skin-lesions\\valid\\melanoma\\*.jpg', 'melanoma')
valid_nevus, valid_nevus_labels = load_images('..\\data\\skin-lesions\\valid\\nevus\\*.jpg', 'nevus')
valid_seborrheic_keratosis, valid_seborrheic_keratosis_labels = load_images('..\\data\\skin-lesions\\valid\\seborrheic_keratosis\\*.jpg', 'seborrheic_keratosis')

all_valid_images = valid_melanoma + valid_nevus + valid_seborrheic_keratosis
all_valid_labels = valid_melanoma_labels + valid_nevus_labels + valid_seborrheic_keratosis_labels

print(len(all_train_images))    
print(len(all_train_labels))

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

