import os
import glob
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
                all_images.append(img)
                labels.append(subdir)  # using the subdirectory name as label

    return all_images, labels




