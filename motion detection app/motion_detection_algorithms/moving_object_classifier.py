import os
import glob
import cv2
import pickle
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.plotting import Annotator

class Detect:
    def __init__(self, video_path, frame_size):
        self.frame_size = frame_size
        self.video_path = video_path

        model_files = glob.glob("models/*")
        labels_path = glob.glob("language labels/*")
        
        device = 'cuda' if torch.cuda.is_available else 'cpu'

        self.model = YOLO(model_files[3])

        self.model.to(device)

        self.labels = self.model.names

        self.cc = torch.randint(0, 255, size=(len(self.labels), 3)).to(device)

        self.detect_classes = set(["person", "bicycle", "car", "motorcycle", "airplane",
                                   "bus", "train", "truck", "boat", "bird", "cat", "dog",
                                   "horse", "sheep", "cow", "bear"])
        
        self.turkish_labels = None
        with open(labels_path[1], 'rb') as file:
            self.turkish_labels = pickle.load(file)

        self.back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)
        self.thresh = 25

        self.cap = cv2.VideoCapture(self.video_path)
    
    def ret_frames(self):
        ret, frame = self.cap.read()
        if ret==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
            fg_mask = self.back_sub.apply(frame)
            results = self.model.predict(frame, verbose=False)

            for r in results:
                annot = Annotator(frame, pil=True)

                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0]
                    b_int = b.type(torch.int)
                    c = box.cls
                    if self.labels[int(c)] in self.detect_classes:
                        if fg_mask[b_int[1]:b_int[3], b_int[0]:b_int[2]].mean() > self.thresh:
                            annot.box_label(b, self.turkish_labels[int(c)], color=(int(self.cc[int(c)][0]),
                                                                                   int(self.cc[int(c)][1]), 
                                                                                   int(self.cc[int(c)][2])))
            
            img = annot.result()

            return [frame, fg_mask, img]
        
        else:
            return False