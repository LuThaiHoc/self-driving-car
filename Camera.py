from __future__ import print_function
import cv2
import numpy as np
import config as cf
import time
import datetime
from threading import Thread



cap = cv2.VideoCapture(0)
#vs = WebcamVideoStream(src=0).start()

def get_frame():
    print("Get_frame started!")
    cap.set(3, 320)
    cap.set(4, 240)
    while cf.running:
        _, frame = cap.read()
        #frame = vs.read()
        frame = cv2.flip(frame,-1)
        #resized = cv2.resize(frame, (480,320), interpolation = cv2.INTER_AREA)
        frame = frame[80:240,:]
        cf.img = frame
        cf.syn = True
        #cv2.imshow("Get_from_camera", cf.img)
        #print(cf.img.shape)
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     break
    #vs.stop()
    cap.release()
    print("Get_frame stoped")

# get_frame()