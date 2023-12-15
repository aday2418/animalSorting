import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
import numpy as np
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class predictiveModel:

    def __init__(self, filePath, objectName):
        self.drawing = filePath
        self.model = self.load_model()
        self.index = self.find_class_index(objectName)

    """def load_model(self):
        return MobileNetV2(weights='imagenet')"""
    
    def load_model(self, num_classes):
        # Load MobileNetV2 with ImageNet weights as a base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model
        base_model.trainable = False

        # Add custom layers on top
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

        return model

    def train_model(self, train_data, val_data, epochs=10, batch_size=32):
        self.model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size
        )
        return history

    def predict_animal(self):
        img = image.load_img(self.drawing, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = self.model.predict(img_array)
        specific_score = predictions[0][self.index]  # Extract the score using the category index
        return specific_score * 100  # Convert to percentage
    
    #The below code returns the top 10 similar scores
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
    
def strokes_to_image(strokes, canvas_dim=256):
    # Create a blank canvas
    image = Image.new("P", (canvas_dim, canvas_dim), color=255)
    draw = ImageDraw.Draw(image)

    # Draw each stroke
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=0, width=2)

    return np.array(image)

def load_data(json_file, label, canvas_dim=256):
    images = []
    labels = []
    with open(json_file, 'r') as file:
        for line in file:
            item = json.loads(line)
            if item['recognized']:  # Optionally, only use recognized drawings
                image = strokes_to_image(item['drawing'], canvas_dim)
                images.append(image)
                labels.append(label)

    return np.array(images), np.array(labels)  

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