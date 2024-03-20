import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from ultralytics import settings
from ultralytics.utils.plotting import Annotator
import pickle
import winsound
import threading

device = 'cuda' if torch.cuda.is_available else 'cpu'

model_files = glob.glob("pt models/yolov8s_epoch50/*")
#sound_files = glob.glob('sounds/*')
video_files = glob.glob('video_footages/*')


model = YOLO(model_files[0])
model.to(device)
labels = model.names

cc = torch.randint(0, 255, size=(len(labels), 3)).to(device)

detect_classes = set(["person", "bicycle", "car", "motorcycle", "airplane",
                      "bus", "train", "truck", "boat", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "bear"])

file_path = 'turkish_labels_2.pickle'
turkish_labels = None
with open(file_path, 'rb') as file:
    turkish_labels = pickle.load(file)



def play_sound(freq):
    winsound.Beep(freq, 500)

back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)
thresh=25


if __name__ == '__main__':
    for video in video_files:
        cap = cv2.VideoCapture(video)
        ret, frame = cap.read()
        h, w, l = frame.shape
        n_h = 360 #int(h/1.5)
        n_w = 600 #int(w/1.5)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                detected = []
                frame = cv2.resize(frame, (n_w, n_h))
                fg_mask = back_sub.apply(frame)
                results = model.predict(frame, verbose=False)
                for r in results:
                    annotator = Annotator(frame, pil=True)
                    
                    boxes = r.boxes
                    for box in boxes:
                        b = box.xyxy[0]
                        b_int = b.type(torch.int)
                        c = box.cls
                        if labels[int(c)] in detect_classes:
                            if fg_mask[b_int[1]:b_int[3], b_int[0]:b_int[2]].mean() > thresh:
                                annotator.box_label(b, turkish_labels[int(c)], color=(int(cc[int(c)][0]),
                                                                                    int(cc[int(c)][1]),
                                                                                    int(cc[int(c)][2])))
                                detected.append(labels[int(c)])
                            

                img = annotator.result()  
                cv2.imshow('YOLO V8 Detection', img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                if 'person' in detected:
                    threading.Thread(target=play_sound, args=[440]).start()
                if "car" in detected:
                    threading.Thread(target=play_sound, args=[260]).start()

            else:
                break
            
        cap.release()
            
        cv2.destroyAllWindows()