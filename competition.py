import random
import tkinter as tk
from tkinter import simpledialog
from tkinter import colorchooser
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

def create_drawing_window(user, filename, selected_animal):
    def paint(event):
        x, y = event.x, event.y
        if eraser_on.get():
            action = "white"
        else:
            action = brush_color.get()
        canvas.create_oval(x - brush_size.get(), y - brush_size.get(), 
                           x + brush_size.get(), y + brush_size.get(), 
                           fill=action, outline=action, width=brush_size.get() * 2)

    def change_color():
        color = colorchooser.askcolor(color=brush_color.get())[1]
        if color:
            brush_color.set(color)

    root = tk.Tk()
    root.title(f"{user}: The selected drawing is {selected_animal}")

    brush_color = tk.StringVar(value="black")
    brush_size = tk.IntVar(value=5)
    eraser_on = tk.BooleanVar(value=False)

    canvas = tk.Canvas(root, width=500, height=500, bg='white')
    canvas.pack(expand=tk.YES, fill=tk.BOTH)
    canvas.bind("<B1-Motion>", paint)

    eraser_button = tk.Button(root, text="Eraser", command=lambda: eraser_on.set(not eraser_on.get()))
    eraser_button.pack(side=tk.BOTTOM)

    # Brush size selection
    brush_size_frame = tk.Frame(root)
    brush_size_frame.pack(side=tk.LEFT, fill=tk.Y)
    tk.Label(brush_size_frame, text="Brush Size").pack()
    for size in [1, 2, 5, 10, 20]:
        button = tk.Button(brush_size_frame, text=str(size), 
                           command=lambda s=size: brush_size.set(s))
        button.pack(side=tk.TOP)

    # Color selection button
    color_button = tk.Button(root, text="Choose Color", command=change_color)
    color_button.pack(side=tk.LEFT)

    # Save and Close button
    button = tk.Button(root, text="Save and Close", command=root.destroy)
    button.pack(side=tk.BOTTOM)

    root.mainloop()


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

    create_drawing_window(user1, file1, selected_animal)
    create_drawing_window(user2, file2, selected_animal)

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