import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (224, 224))
    preprocessed_img = resized_img / 255.0 
    return np.expand_dims(preprocessed_img, axis=0)


model_path = "./best_model.h5" 
loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})


testing_folder = "./testing"  

testing_files = os.listdir(testing_folder)


for testing_file in testing_files:
    
    testing_image_path = os.path.join(testing_folder, testing_file)
    preprocessed_image = load_and_preprocess_image(testing_image_path)
    predictions = loaded_model.predict(preprocessed_image)
    predicted_label = np.argmax(predictions)
    event_labels_dict = {
        0:"combat",
        1:"destroyedbuilding",
        2:"humanitarianaid",
        3:"militaryvehicles",
        4:"fire",
    }
    print(f"Image: {testing_file}, Predicted Label: {event_labels_dict[predicted_label]}")
