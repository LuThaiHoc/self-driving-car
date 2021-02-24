import cv2
import numpy as np
import config as cf
import time
cap = cv2.VideoCapture("test2.mp4")

def get_frame():
    print("Get_frame started!")
    # cap.set(3, 480)
    # cap.set(4, 320)
    while cf.running:
        _, frame = cap.read()
        #frame = cv2.flip(frame,-1)
        resized = cv2.resize(frame, (480,320), interpolation = cv2.INTER_AREA)
        cf.img = resized
        cf.syn = True
        #cv2.imshow("Get_from_camera", cf.img)
        #print(cf.img.shape)
        # k = cv2.waitKey(1)
        # if k == ord('q'):
        #     break
    cap.release()
    print("Get_frame stoped")

# get_frame()