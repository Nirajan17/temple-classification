import tensorflow as tf


def parse_tfrecord_fn(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'class': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.image.decode_jpeg(example['image'], channels=3)  # Modify the decoding function as per image format
#     image = tf.cast(image, tf.float32) / 255.0
    
    label = tf.cast(example['class'], tf.string)
    
    return image, label


# Specify the path to the TFRecord file(s)
tfrecord_files = ['/Users/nirajanpaudel17/Documents/Python/Major-Project/Temple-Classification/labeled_temples_images.tfrecord']

# Create a TFRecordDataset
dataset = tf.data.TFRecordDataset(tfrecord_files)

# Apply parsing function to the dataset
dataset = dataset.map(parse_tfrecord_fn)

# Print the first few examples in the dataset
for image, label in dataset.take(1):
    # Process or use the image and label as needed
    print(image.shape, label)


# # splitting the datset into train and test set

# dataset = dataset.shuffle(buffer_size=1000, seed=42)

# # Split the dataset into train and test sets

# num_samples = int(dataset.cardinality().numpy())

# train_size = int(0.8 * num_samples)  # 80% for training
# test_size = dataset.cardinality() - train_size

# train_dataset = dataset.take(train_size)
# test_dataset = dataset.skip(train_size)

# # Print the number of samples in each set
# print("Train set size:", train_size)
# print("Test set size:", test_size)