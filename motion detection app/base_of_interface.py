import customtkinter
from PIL import Image
import os
import numpy as np

import sof_interface
import dof_interface

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

class OfInterface(customtkinter.CTkToplevel):
    def __init__(self, master, path, img_size):
        super().__init__(master)

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.path = path
        self.img_size = img_size

        self.button1 = ButtonFrame(self, "Sparse Optical Flow", "images/sparse_optical_flow_template.jpg",
                                   self.img_size,
                                    0.6,
                                    command = lambda : self.open_sparse_interface(self.path, self.img_size))

        self.button2 = ButtonFrame(self, "Dense Optical Flow", "images/dense_optical_flow_template.jpg",
                                   self.img_size,
                                   0.6,
                                   command = lambda : self.open_dense_interface(self.path, self.img_size))

        self.button1.grid(row=0, column=0)
        self.button2.grid(row=0, column=1)

        self.exit_button = customtkinter.CTkButton(self, text="Close", font=('Consolas', 16), command=self.close)
        self.exit_button.grid(row=1, column=0, padx=10, pady=10, columnspan=2, sticky='s')

    def close(self):
        self.destroy()
        self.update()

    def open_sparse_interface(self, path, img_size):
        window = sof_interface.Video_interface(self, path, img_size)
        window.title("sparse optical flow")

    def open_dense_interface(self, path, img_size):
        window = dof_interface.Video_interface(self, path, img_size)
        window.title("dense optical flow")