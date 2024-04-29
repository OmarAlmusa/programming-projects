import customtkinter
import glob
from PIL import Image
import numpy as np
import os

import fd_interface
import bgsub_interface

customtkinter.set_appearance_mode('dark')

class LabelFrame(customtkinter.CTkFrame):
    def __init__(self, master, label, img_path, img_size, command):
        super().__init__(master)
        """
            img_size = list[width(int), height(int)]
        
        """

        self.label = label
        self.pil_image = Image.open(os.path.join(img_path)).resize((img_size[0], img_size[1]))
        self.pil_image = 0.6*np.ones((img_size[1], img_size[0], 1)) * self.pil_image
        self.img = customtkinter.CTkImage(Image.fromarray(np.uint8(self.pil_image)), size=(img_size[0], img_size[1]))

        self.button = customtkinter.CTkButton(self, image=self.img, text=label, font=("Consolas", 14),
                                              compound='top', fg_color='transparent', bg_color='transparent',
                                              corner_radius=30, hover_color='#702963', border_color='#FFC000', border_width=2, command=command)
        self.button.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")
        
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.video_path = glob.glob('videos/*')
        self.img_size = [600, 300]
        self.label1 = LabelFrame(self, "Frame Difference", "images/frame_difference_template.jpg",
                                 self.img_size, 
                                 command=lambda : self.open_fd_interface(self.video_path[0], self.img_size))
        
        self.label2 = LabelFrame(self, "Background Subtraction", "images/background_subtraction_template.jpg", 
                                 self.img_size,
                                 command=lambda : self.open_bgsub_interface(self.video_path[0], self.img_size))
        
        self.label3 = LabelFrame(self, "Optical Flow", "images/optical_flow_template.jpg", self.img_size, command=None)
        self.label4 = LabelFrame(self, "Moving Object Classifier", "images/moving_object_classifier_template.jpg", self.img_size, command=None)

        self.label1.grid(row=0, column=0)
        self.label2.grid(row=0, column=1)
        self.label3.grid(row=1, column=0)
        self.label4.grid(row=1, column=1)

        self.button_exit = customtkinter.CTkButton(self, text="Exit", font=("Consolas", 14),
                                                   compound='top', fg_color='transparent', bg_color='transparent',
                                                   corner_radius=30, hover_color='#702963', border_color='#FFC000', border_width=2, command=self.quit)
        self.button_exit.grid(row=2, column=0, padx=10, pady=10, columnspan=2, sticky='s')

    def open_fd_interface(self, path, img_size):
        window = fd_interface.Video_interface(self, path, img_size)
        window.title("Frame Difference")

    def open_bgsub_interface(self, path, img_size):
        window = bgsub_interface.Video_interface(self, path, img_size)
        window.title("Background Subtraction")


app = App()
app.mainloop()
