import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import json


class predictiveModel:

    def __init__(self, filePath, objectName):
        self.drawing = filePath
        self.model = self.load_model()
        self.index = self.find_class_index(objectName)

    def load_model(self):
        return MobileNetV2(weights='imagenet')

    def predict_animal(self):
        img = image.load_img(self.drawing, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = self.model.predict(img_array)
        specific_score = predictions[0][self.index]  # Extract the score using the category index
        return specific_score * 100  # Convert to percentage
    """img = image.load_img(self.drawing, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = self.model.predict(img_array)
        return decode_predictions(predictions, top=10)[0]"""
    
    def find_class_index(self, class_name):
        # Path to the ImageNet class index file
        class_index_path = 'imagenet_class_index.json'

        # Convert class_name to lowercase for case-insensitive comparison
        class_name_lower = class_name.lower()

        # Load the class index file
        with open(class_index_path) as file:
            class_index = json.load(file)

        # Search for the class
        for index, label in class_index.items():
            # Convert label to lowercase and split to get the individual names
            label_names = label.lower().split(', ')
            if class_name_lower in label_names:
                return int(index)
        print("Not a valid classname")
        return None
    

    
    

"""def main():
    print("starting")
    model = load_model()
    print("Loaded Model")
    img_path = input("Please enter a filename or filepath:")
    predictions = predict_animal(model, img_path)

    print("Predictions:")

    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")

if __name__ == "__main__":
    main()"""