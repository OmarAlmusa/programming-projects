import os
import glob
import cv2
import numpy as np


class Detect:
    def __init__(self, video_path, frame_size):
        self.frame_size = frame_size
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        ret, self.frame1 = self.cap.read()
        self.kernel = np.ones((5, 5), np.uint8)

        self.frame1 = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2RGB)
        self.frame1 = cv2.resize(self.frame1, (self.frame_size[0], self.frame_size[1]))

        self.prvs = cv2.cvtColor(self.frame1, cv2.COLOR_RGB2GRAY)
        self.hsv = np.zeros_like(self.frame1)
        self.hsv[..., 1] = 255

    def get_motion_mask(self, flow_mag, motion_thresh=1, kernel=np.ones((7, 7))):
        
        motion_mask = np.uint8(flow_mag > motion_thresh)*255

        motion_mask =  cv2.erode(motion_mask, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        return motion_mask
        
        
    def ret_frames(self):
        
    
        ret, frame2 = self.cap.read()
        if ret == True:
            
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame2 = cv2.resize(frame2, (self.frame_size[0], self.frame_size[1]))
            next_ = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(self.prvs, next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            self.hsv[..., 0] = ang*180/np.pi/2
            self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2RGB)

            fx = self.get_motion_mask(mag)
            bbox_frame = frame2.copy()
            contours, _ = cv2.findContours(fx, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                area = w*h
                if area>200:
                    cv2.rectangle(bbox_frame, (x, y), (x+w, y+h), (0, 255, 0), thickness = 2)

            self.prvs = next_


            return [frame2, bgr, fx, bbox_frame]
        
        else:
            return False