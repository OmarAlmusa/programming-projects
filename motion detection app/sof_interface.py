import motion_detection_algorithms.sparse_optical_flow as sparse_optical_flow
import customtkinter
from PIL import Image

class Video_interface(customtkinter.CTkToplevel):
    def __init__(self, master, video_path, frame_size):
        super().__init__(master)

        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.frame_size = frame_size
        self.video = video_path
        self.tn = 10
        self.detections = sparse_optical_flow.Detect(self.video, self.frame_size)

        self.button1 = customtkinter.CTkButton(self, text="start video", font=('Consolas', 16), command=self.start_video)
        self.button1.grid(row=0, column=0, padx=10, pady=10, columnspan=2, sticky='n')

        self.label_widget1 = customtkinter.CTkLabel(self, text="Original", compound='top')
        self.label_widget1.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

        self.label_widget2 = customtkinter.CTkLabel(self, text="Detections", compound='top')
        self.label_widget2.grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

        self.exit_button = customtkinter.CTkButton(self, text='Close', font=('Consolas', 16), command=self.close)
        self.exit_button.grid(row=2, column=0, padx=10, pady=10, columnspan=2, sticky='s')

    def close(self):
        self.destroy()
        self.update()


    def start_video(self):
        
        detections = self.detections.ret_frames()
        if detections != False:
            frame, points_img = detections

            captured_image1 = Image.fromarray(frame)
            tk_image1 = customtkinter.CTkImage(captured_image1, size=(self.frame_size[0], self.frame_size[1]))
            self.label_widget1.configure(image=tk_image1)

            captured_image2 = Image.fromarray(points_img)
            tk_image2 = customtkinter.CTkImage(captured_image2, size=(self.frame_size[0], self.frame_size[1]))
            self.label_widget2.configure(image=tk_image2)

            
            self.label_widget1.after(self.tn, self.start_video)

        else:
            self.detections = sparse_optical_flow.Detect(self.video, self.frame_size)

            self.label_widget1.after(self.tn, self.start_video)