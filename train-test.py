import os
import random
import shutil


source_folder = '/Users/nirajanpaudel17/Documents/Python/Major-Project/Web-Scrapping/images'
destination_folder = '/Users/nirajanpaudel17/Documents/Python/Major-Project/Temple-Classification/dataset'

train_ratio = 0.7  # 70% for training, 30% for testing

for category_folder in os.listdir(source_folder):
    category_path = os.path.join(source_folder, category_folder)
    if not os.path.isdir(category_path):
        continue  # Skip non-directory files
    
    # Create corresponding folders in the destination folder
    train_category_path = os.path.join(destination_folder, 'train', category_folder)
    test_category_path = os.path.join(destination_folder, 'test', category_folder)
    os.makedirs(train_category_path, exist_ok=True)
    os.makedirs(test_category_path, exist_ok=True)
    
    # Collect image file paths in the current category
    image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Randomly shuffle the image file paths

    random.shuffle(image_files)
    
    # Determine the split index based on the train_ratio
    split_index = int(len(image_files) * train_ratio)
    
    # Move images to the train or test folder based on the split index
    for i, image_file in enumerate(image_files):
        source_path = os.path.join(category_path, image_file)
        if i < split_index:
            destination_path = os.path.join(train_category_path, image_file)
        else:
            destination_path = os.path.join(test_category_path, image_file)
        
        # Move the image file
        shutil.move(source_path, destination_path)
