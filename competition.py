import random
import tkinter as tk
from tkinter import simpledialog
from PIL import ImageGrab
from predictAnimal import predictiveModel


# List of animals
animals = ["lion", "tiger", "bear", "elephant", "giraffe"]

# Function to randomly select an animal
def choose_animal():
    return random.choice(animals)

def save_canvas(canvas, filename):
    # Calculate the position of the canvas
    x = canvas.winfo_rootx() + canvas.winfo_x()
    y = canvas.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Grab the image and save
    ImageGrab.grab(bbox=(x, y, x1, y1)).save(filename)

# Placeholder for drawing submission (In a real application, this would be more complex)
def create_drawing_window(user):
    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    root = tk.Tk()
    root.title(f"Drawing Window for {user}")

    canvas = tk.Canvas(root, width=500, height=500, bg='white')
    canvas.pack(expand=tk.YES, fill=tk.BOTH)
    canvas.bind("<B1-Motion>", paint)

    # Function to handle the save operation
    def save_and_close():
        save_canvas(canvas, f"drawing.png")
        root.destroy()

    # Button to save the drawing and close the window
    button = tk.Button(root, text="Save and Close", command=save_and_close)
    button.pack(side=tk.BOTTOM)

    root.mainloop()
# Placeholder for drawing comparison (This would be a complex image processing task)
def compare_drawings(drawing1):
    prediction = predictiveModel(drawing1)
    return prediction.predict_animal()



# Main program
def predictAnimal():
    user1 = "User 1"
    user2 = "User 2"

    selected_animal = choose_animal()
    print(f"The selected animal is: {selected_animal}")

    #create_drawing_window(user1)

    predictions = compare_drawings("drawing.png")
    
    for i, (imagenet_id, label, score) in enumerate(predictions):
        print(f"{i + 1}: {label} ({score * 100:.2f}%)")
        if label == 'lion':
            return score * 100

    #print(f"The winner is: {winner}")

if __name__ == "__main__":
    predictAnimal()