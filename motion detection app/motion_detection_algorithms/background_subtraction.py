import os
import glob
import cv2
import numpy as np


class Detect:
    def __init__(self, video_path, frame_size):
        self.frame_size = frame_size
        self.video_path = video_path
        self.backSubMOG = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        self.area_thresh = 500
        self.cap = cv2.VideoCapture(self.video_path)
        
    def ret_frames(self):
    
        ret, frame = self.cap.read()
        if ret == True:
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
            blur = cv2.GaussianBlur(frame, (11, 11), 0)
            fgMaskMOG = self.backSubMOG.apply(blur)

            contours, hierarchy = cv2.findContours(fgMaskMOG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_frame = frame.copy()
            for contour in contours:

                # get the xmin, ymin, width, and height coordinates from the contours
                (x, y, w, h) = cv2.boundingRect(contour)
                area = w*h
                # draw the bounding boxes
                if area > self.area_thresh:
                    cv2.rectangle(bbox_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


            return [frame, fgMaskMOG, bbox_frame]
        
        else:
            return False