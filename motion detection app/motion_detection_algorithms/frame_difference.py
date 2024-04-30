import os
import glob
import cv2
import numpy as np


class Detect:
    def __init__(self, video_path, frame_size):
        self.frame_size = frame_size
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        ret, self.old_frame = self.cap.read()
        self.old_frame = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2RGB)
        self.old_frame = cv2.resize(self.old_frame, (self.frame_size[0], self.frame_size[1]))
        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_RGB2GRAY)
        
    def ret_frames(self):
        
        ret, frame = self.cap.read()
        if ret == True:
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
            delta_frame = cv2.absdiff(gray, self.old_gray)
            Mn = cv2.threshold(delta_frame, 35, 255, cv2.THRESH_BINARY)[1]

            contours, hierarchy = cv2.findContours(Mn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_frame = frame.copy()
            for contour in contours:
                if cv2.contourArea(contour) < 25:
                    continue

                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv2.boundingRect(contour)
                # draw the bounding boxes
                cv2.rectangle(bbox_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.old_gray = gray.copy()

            return [frame, delta_frame, Mn, bbox_frame]
        
        else:
            return False