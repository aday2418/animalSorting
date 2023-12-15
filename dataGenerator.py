import numpy as np
import json
from tensorflow.keras.utils import Sequence, to_categorical  # Import to_categorical
from PIL import Image, ImageDraw

class DoodleDataGenerator(Sequence):
    def __init__(self, data, label, batch_size, canvas_dim=256, shuffle=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.canvas_dim = canvas_dim
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_json(self):
        with open(self.json_file, 'r') as file:
            return [json.loads(line) for line in file]

    def _strokes_to_image(self, strokes):
        image = Image.new("P", (self.canvas_dim, self.canvas_dim), color=255)
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            for i in range(len(stroke[0]) - 1):
                draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=0, width=2)
        return np.array(image)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        # Calculate start and end indices for the batch
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.data))  # Adjust for the last batch

        # Adjust batch size for the last batch
        current_batch_size = end_idx - start_idx

        # Initialize arrays for storing batch data
        batch_x = np.empty((current_batch_size, self.canvas_dim, self.canvas_dim, 1), dtype=np.float32)
        batch_y = np.empty((current_batch_size, 2), dtype=np.float32)  # Assuming 2 classes

        # Process each item in the batch
        for i, idx in enumerate(range(start_idx, end_idx)):
            item = self.data[idx]
            if item['recognized']:
                image = self._strokes_to_image(item['drawing'])
                batch_x[i,] = np.expand_dims(image, axis=-1) / 255.0  # Normalize the image
                batch_y[i,] = to_categorical(self.label, num_classes=2)

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)