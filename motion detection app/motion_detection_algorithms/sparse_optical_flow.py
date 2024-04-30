import os
import glob
import cv2
import numpy as np


class Detect:
    def __init__(self, video_path, frame_size):
        self.frame_size = frame_size
        self.video_path = video_path
        self.feature_params = dict(maxCorners= 100, qualityLevel = 0.3,
                                   minDistance=7, blockSize=7)
        
        self.lk_params = dict(winSize=(15, 15), maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (100, 3))
        self.cap = cv2.VideoCapture(self.video_path)

        ret, self.old_frame = self.cap.read()
        self.old_frame = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2RGB)
        self.old_frame = cv2.resize(self.old_frame, (self.frame_size[0], self.frame_size[1]))
        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_RGB2GRAY)
        self.prev = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

        self.mask = np.zeros_like(self.old_frame)
        
    def ret_frames(self):
        
    
        ret, frame = self.cap.read()
        if ret == True:
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            #calculate optical flow
            next_, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray,
                                                           self.prev, None, 
                                                           **self.lk_params)
            good_old = self.prev[status == 1]
            good_new = next_[status == 1]

            img = frame.copy()
            #draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                img = cv2.circle(img, (int(a), int(b)), 3, self.color[i].tolist(), -1)
            
            img = cv2.add(img, self.mask)

            self.old_gray = frame_gray.copy()
            self.prev = good_new.reshape(-1, 1, 2)

            return [frame, img]
        
        else:
            return False