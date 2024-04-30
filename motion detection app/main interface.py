import customtkinter
from PIL import Image
import numpy as np
import os

import fd_interface
import bgsub_interface
import base_of_interface
import moc_interface

customtkinter.set_appearance_mode('dark')

class ButtonFrame(customtkinter.CTkFrame):
    def __init__(self, master, label, img_path, img_size, mul_value, command):
        super().__init__(master)
        """
            img_size = list[width(int), height(int)]
        
        """

        pil_image = Image.open(os.path.join(img_path)).resize((img_size[0], img_size[1]))
        pil_image = mul_value*np.ones((img_size[1], img_size[0], 1)) * pil_image
        img = customtkinter.CTkImage(Image.fromarray(np.uint8(pil_image)), size=(img_size[0], img_size[1]))

        button = customtkinter.CTkButton(self, image=img, text=label, font=("Consolas", 14),
                                         compound='top', fg_color='transparent', bg_color='transparent',
                                         corner_radius=30, hover_color='#702963', command=command)
        button.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")
        
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.img_size = [600, 300]
        self.label1 = ButtonFrame(self, "Frame Difference", "images/frame_difference_template.jpg",
                                 self.img_size, 
                                 0.6,
                                 command=lambda : self.open_fd_interface(self.img_size))
        
        self.label2 = ButtonFrame(self, "Background Subtraction", "images/background_subtraction_template.jpg", 
                                 self.img_size,
                                 0.6,
                                 command=lambda : self.open_bgsub_interface(self.img_size))
        
        self.label3 = ButtonFrame(self, "Optical Flow", "images/optical_flow_template.jpg",
                                 self.img_size, 
                                 0.6,
                                 command=lambda : self.open_baseof_interface(self.img_size))
        
        self.label4 = ButtonFrame(self, "Moving Object Classifier", "images/moving_object_classifier_template.jpg",
                                  self.img_size, 
                                  0.6,
                                  command=lambda : self.open_moc_interface(self.img_size))

        self.label1.grid(row=0, column=0)
        self.label2.grid(row=0, column=1)
        self.label3.grid(row=1, column=0)
        self.label4.grid(row=1, column=1)

        self.button_exit = customtkinter.CTkButton(self, text="Exit", font=("Consolas", 14),
                                                   compound='top', fg_color='#555555', bg_color='transparent',
                                                   corner_radius=30, hover_color='#702963', command=self.quit)
        self.button_exit.grid(row=2, column=0, padx=10, pady=10, columnspan=2, sticky='s')

    def open_fd_interface(self, img_size):
        window = fd_interface.Video_interface(img_size)
        window.title("Frame Difference")

    def open_bgsub_interface(self, img_size):
        window = bgsub_interface.Video_interface(img_size)
        window.title("Background Subtraction")

    def open_baseof_interface(self, img_size):
        window = base_of_interface.OfInterface(img_size)
        window.title("Optical Flow")
    
    def open_moc_interface(self, img_size):
        window = moc_interface.Video_interface(img_size)
        window.title("Moving Object Classifier")


app = App()
app.mainloop()
