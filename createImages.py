import pyautogui
import keyboard
import re
import tkinter as tk
from PIL import Image, ImageTk
import os

buttons = ['left', 'right', 'mid']


def show_popup():
    screenshot = pyautogui.screenshot(region=(1079, 19, 261, 200)) # Захватываем пиксели вебки

    screenshot.save('untitle.png')
    
    popup = tk.Toplevel()
    popup.title("Всплывающее окно")
    popup.attributes('-topmost', True)
    
    image = Image.open('untitle.png')
    photo = ImageTk.PhotoImage(image)
    
    label = tk.Label(popup, image=photo)
    label.image = photo
    label.pack()
    
    frame = tk.Frame(popup)
    frame.pack()
    button_objects = []
    for i in range(len(buttons)):
        button = tk.Button(frame, text=buttons[i], command=lambda i=i, window=popup: button_pressed(i, window))
        button_objects.append(button)
        button.grid(row=0, column=i)
    popup.protocol("WM_DELETE_WINDOW", lambda: destroy_popup(popup))
    popup.mainloop()

def destroy_popup(popup):
    popup.destroy()
    root.deiconify()

def button_pressed(index, window):
    IMAGE_DIR = os.path.dirname(__file__)
    arr_images_names = os.listdir(IMAGE_DIR)
    max_number = 0

    for filename in arr_images_names:
        match = re.match(r'^(\d+)', filename) 
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    os.replace('untitle.png', str(max_number + 1) + buttons[index] + '.png')
    window.destroy()

root = tk.Tk()
root.withdraw()

def handle_w_key():
    root.after(1, lambda: show_popup())  

keyboard.add_hotkey('`', handle_w_key) # здесь можно указать удобную кнопку для скрина

root.mainloop()
