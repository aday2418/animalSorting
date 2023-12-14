import random
import tkinter as tk
from tkinter import simpledialog
from PIL import ImageGrab
from predictAnimal import predictiveModel
import json


# List of animals
def load_animal_names(json_path):
    #print(json_path)
    with open(json_path, 'r') as file:
        data = json.load(file)
        animal_list = [label.split(', ')[0] for index, label in data.items()]
        return animal_list
    
def choose_animal(json_path):
    #print(json_path)
    return random.choice(load_animal_names(json_path))

def save_canvas(canvas, filename):
    # Calculate the position of the canvas
    x = canvas.winfo_rootx() + canvas.winfo_x()
    y = canvas.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Grab the image and save
    ImageGrab.grab(bbox=(x, y, x1, y1)).save(filename)

# Placeholder for drawing submission (In a real application, this would be more complex)
def create_drawing_window(user, filename):
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
        save_canvas(canvas, filename)
        root.destroy()

    # Button to save the drawing and close the window
    button = tk.Button(root, text="Save and Close", command=save_and_close)
    button.pack(side=tk.BOTTOM)

    root.mainloop()
# Placeholder for drawing comparison (This would be a complex image processing task)
def get_score(drawing1, selected_animal):
    prediction = predictiveModel(drawing1, selected_animal)
    return prediction.predict_animal()

def main():
    user1 = "User 1"
    user2 = "User 2"

    selected_animal = choose_animal("imagenet_class_index.json")
    print(f"The selected animal is: {selected_animal}")

    file1 = "user1.png"
    file2 = "user2.png"

    create_drawing_window(user1, file1)
    create_drawing_window(user2, file2)

    likenessScore1 = get_score(file1, selected_animal)
    likenessScore2 = get_score(file2, selected_animal)
   
    print(f"User1 -- {selected_animal} likeness score: {likenessScore1}%")
    print(f"User2 -- {selected_animal} likeness score: {likenessScore2}%")
    if (likenessScore1 > likenessScore2):
        print("User 1 wins!!")
    else:
        print("User 2 wins!!")

if __name__ == "__main__":
    main()