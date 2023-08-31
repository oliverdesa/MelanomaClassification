import os
import glob
import numpy as np
from keras.preprocessing import image

def load_images(main_directory):
    all_images = []
    labels = []

    # Loop through each subdirectory in the main directory
    for subdir in os.listdir(main_directory):
        full_subdir_path = os.path.join(main_directory, subdir)

        # Check if it's a directory (not a file or other type)
        if os.path.isdir(full_subdir_path):

            # Loop through each file in the subdirectory
            for file in glob.glob(os.path.join(full_subdir_path, '*.*')):
                img = image.load_img(file, target_size=(224, 224))
                img_array = image.img_to_array(img)
                all_images.append(img_array)
                labels.append(subdir)  # using the subdirectory name as label
    
    np_images = np.array(all_images).astype('float32')  # Convert to numpy array
    np_images /= 255.0 # Normalize images

    return np_images, labels




