# creating tfrecord file

import tensorflow as tf
import os
from matplotlib.image import imread

# Define the paths to your image folders
folder_paths = {
    'Bhaktapur-Durbar-Square': '/Users/nirajanpaudel17/Documents/Python/Major-Project/Web-Scrapping/images/Bhaktapur-Durbar-Square',
    'Bouddanath': '/Users/nirajanpaudel17/Documents/Python/Major-Project/Web-Scrapping/images/Bouddanath',
    'Pashupatinath': '/Users/nirajanpaudel17/Documents/Python/Major-Project/Web-Scrapping/images/Pashupatinath',
    'Patan-Durbar-Square': '/Users/nirajanpaudel17/Documents/Python/Major-Project/Web-Scrapping/images/Patan-Durbar-Square',
    'Swyambunath': '/Users/nirajanpaudel17/Documents/Python/Major-Project/Web-Scrapping/images/Swyambunath'
}

# Define the output TFRecord file name
output_directory = '/Users/nirajanpaudel17/Documents/Python/Major-Project/Temple-Classification'
output_filename = os.path.join(output_directory, 'labeled_temples_images.tfrecord')

# Initialize an empty list to store the labeled images
labeled_images = []

# Iterate over each class folder
for class_name, folder_path in folder_paths.items():
    
    # Get the list of image filenames in the current class folder
    image_filenames = os.listdir(folder_path)
    

    # Iterate over each image in the class folder
    for image_filename in image_filenames:
        # Create the full path to the image file
        image_path = os.path.join(folder_path, image_filename)

        try:
            # Read the image file
            image = tf.io.read_file(image_path)

            # Decode the image file
            image = tf.io.decode_jpeg(image)
            
            image = tf.io.encode_jpeg(image).numpy()


            # Create a labeled example
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'class': tf.train.Feature(bytes_list=tf.train.BytesList(value=[class_name.encode()]))
            }))

            # Append the labeled example to the list
            labeled_images.append(example)

        except tf.errors.InvalidArgumentError:
            print('Skipping unsupported image:', image_path)

# Create a writer for the TFRecord file
with tf.io.TFRecordWriter(output_filename) as writer:
    # Write each labeled example to the TFRecord file
    for example in labeled_images:
        writer.write(example.SerializeToString())

# # Print the number of labeled images
print('Number of labeled images:', len(labeled_images))
