import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(physical_devices) > 0:
    print("GPU available, enabling memory growth.")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("GPU not available, using CPU.")

data_dir = './training'  # Update this with the actual path to your dataset

event_images_dict = {
    'combat': [os.path.join(data_dir, 'Combat', file) for file in os.listdir(os.path.join(data_dir, 'Combat'))],
    'destroyedbuilding': [os.path.join(data_dir, 'DestroyedBuildings', file) for file in os.listdir(os.path.join(data_dir, 'DestroyedBuildings'))],
    'humanitarianaid': [os.path.join(data_dir, 'Humanitarian Aid and rehabilitation', file) for file in os.listdir(os.path.join(data_dir, 'Humanitarian Aid and rehabilitation'))],
    'militaryvehicles': [os.path.join(data_dir, 'Military vehicles and weapons', file) for file in os.listdir(os.path.join(data_dir, 'Military vehicles and weapons'))],
    'fire': [os.path.join(data_dir, 'Fire', file) for file in os.listdir(os.path.join(data_dir, 'Fire'))],
}

event_labels_dict = {
    'combat': 0,
    'destroyedbuilding': 1,
    'humanitarianaid': 2,
    'militaryvehicles': 3,
    'fire': 4,
}

# Optional: Filter out non-image files (e.g., if there are non-image files in the directories)
for category, file_paths in event_images_dict.items():
    event_images_dict[category] = [file_path for file_path in file_paths if file_path.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Print the dictionary for verification
for category, file_paths in event_images_dict.items():
    print(f'{category}: {len(file_paths)} images')

X, y = [], []

for event_name, images in event_images_dict.items():
    for image_path in images:
        img = cv2.imread(str(image_path))
        resized_img = cv2.resize(img, (224, 224))
        X.append(resized_img)
        y.append(event_labels_dict[event_name])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

num_of_events = 5

data_augmentation = tf.keras.Sequential([
   tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
   tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
   tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
])


checkpoint_path = "best_model.h5"


checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_acc',
    save_best_only=True, 
    mode='max',
    verbose=1 
)

model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_of_events)
])

model.build((None, 224, 224, 3))
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc']
)

callbacks = [checkpoint_callback]

model.fit(
    X_train_scaled, y_train,
    epochs=10,
    validation_data=(X_test_scaled, y_test),
    callbacks=callbacks
)

