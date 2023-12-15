import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam  # Import Adam optimizer
from sklearn.model_selection import train_test_split
import json
from PIL import Image, ImageDraw
from dataGenerator import DoodleDataGenerator
# Import any other necessary libraries

class ModelTrainer:
    def __init__(self, data_path, canvas_dim=256):
        self.data_path = data_path
        self.canvas_dim = canvas_dim
        self.model = None
    
    def strokes_to_image(self, strokes, canvas_dim=256):
        # Create a blank canvas
        image = Image.new("P", (canvas_dim, canvas_dim), color=255)
        draw = ImageDraw.Draw(image)

        # Draw each stroke
        for stroke in strokes:
            for i in range(len(stroke[0]) - 1):
                draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=0, width=2)

        return np.array(image)
    
    def load_data(self, json_file, label, canvas_dim=256):
        with open(json_file, 'r') as file:
            data = [json.loads(line) for line in file]

        # Split data into training and validation sets
        split_index = len(data) // 2
        train_data = data[:split_index]
        val_data = data[split_index:]

        return train_data, val_data


    """def process_data(self, data, label, canvas_dim):
        images = []
        labels = []
        for item in data:
            if item['recognized']:  # Optionally, only use recognized drawings
                image = self.strokes_to_image(item['drawing'], canvas_dim)
                images.append(image)
                # Append the label for each image
                labels.append(label)

        images = np.array(images).reshape((-1, canvas_dim, canvas_dim, 1)).astype('float32') / 255.0
        # Ensure labels are one-hot encoded
        labels = to_categorical(np.array(labels), num_classes=2)

        return images, labels"""


    def build_model(self):
        # Build the CNN model
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.canvas_dim, self.canvas_dim, 1)),
            MaxPooling2D(2, 2),
            # Add more layers as needed
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')  # Assuming 2 classes
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    def save_model(self, file_name):
        # Save the trained model
        self.model.save(file_name)

    def run(self):
        # Load and preprocess the data
        train_data, val_data = self.load_data(self.data_path, 0)  # Label '0' for bees
        train_generator = DoodleDataGenerator(train_data, label=0, batch_size=32, canvas_dim=self.canvas_dim)
        val_generator = DoodleDataGenerator(val_data, label=0, batch_size=32, canvas_dim=self.canvas_dim)

        self.build_model()
        self.model.fit(train_generator, epochs=5, validation_data=val_generator)

        # Save the model
        self.save_model('trainedDoodleModel.h5')
    
# Usage
if __name__ == "__main__":
    trainer = ModelTrainer(data_path='json/full_simplified_bee.ndjson')
    trainer.run()