import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img

data_dir = './training'  # Update this with the actual path to your dataset

event_images_dict = {
    'combat': [os.path.join(data_dir, 'Combat', file) for file in os.listdir(os.path.join(data_dir, 'Combat'))],
    'destroyedbuilding': [os.path.join(data_dir, 'DestroyedBuildings', file) for file in os.listdir(os.path.join(data_dir, 'DestroyedBuildings'))],
    'humanitarianaid': [os.path.join(data_dir, 'Humanitarian Aid and rehabilitation', file) for file in os.listdir(os.path.join(data_dir, 'Humanitarian Aid and rehabilitation'))],
    'militaryvehicles': [os.path.join(data_dir, 'Military vehicles and weapons', file) for file in os.listdir(os.path.join(data_dir, 'Military vehicles and weapons'))],
    'fire': [os.path.join(data_dir, 'Fire', file) for file in os.listdir(os.path.join(data_dir, 'Fire'))],
}

augmentation_factor = 10  # Increase this factor as needed

for category, file_paths in event_images_dict.items():
    print(f'Augmenting images in {category} folder...')
    
    for file_path in file_paths:
        img = load_img(file_path)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Create a directory to save augmented images
        save_dir = os.path.join(os.path.dirname(file_path))
        os.makedirs(save_dir, exist_ok=True)

        # Apply data augmentation and save augmented images
        datagen = image.ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        i = 0
        for batch in datagen.flow(img_array, save_to_dir=save_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= augmentation_factor:
                break  # Break the loop after creating the desired number of augmented images

print('Data augmentation complete.')
