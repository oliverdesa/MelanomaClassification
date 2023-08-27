#import tensorflow as tf
#print(tf.__version__)
import os
import glob
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import numpy as np

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


def load_images(image_path):
    all_images = []
    for file in glob.glob(image_path):
        img = image.load_img(file, target_size=(224, 224))
        all_images.append(img)
    return all_images

train_melanoma = load_images('..\\..\\data\\skin-lesions\\train\\melanoma\\*.jpg')
print(len(train_melanoma))
# train_nevus = load_images('skin-lesions\\train\\nevus\\*.jpg')
# train_seborrheic_keratosis = load_images('skin-lesions\\train\\seborrheic_keratosis\\*.jpg')    

# test_melanoma = load_images('skin-lesions\\test\\melanoma\\*.jpg')
# test_nevus = load_images('skin-lesions\\test\\nevus\\*.jpg')
# test_seborrheic_keratosis = load_images('skin-lesions\\test\\seborrheic_keratosis\\*.jpg')

# valid_melanoma = load_images('skin-lesions\\valid\\melanoma\\*.jpg')
# valid_nevus = load_images('skin-lesions\\valid\\nevus\\*.jpg')
# valid_seborrheic_keratosis = load_images('skin-lesions\\valid\\seborrheic_keratosis\\*.jpg')

# convert the image pixels to a numpy array
def convert_to_array(all_images):
    all_images_array = []
    for img in all_images:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        all_images_array.append(x)
    return all_images_array




