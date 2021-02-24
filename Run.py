import cv2
import numpy as np
import time
import threading
import sys
#from Modules.CarControl import speed_control, angle_control, PID
#from Modules.DetectLane import get_center
#from Modules.Camera import get_frame
from CarControl import speed_control, angle_control, PID
from getLaneByHSV import getLaneCenter
#from DetectLane import get_center
#from getLaneCenter import getLaneCenter
from Camera import get_frame
import config as cf

# System Variables
cf.HEIGHT = 240
cf.WIDTH = 320
cf.is_record = False
cf.img = np.zeros((320, 160, 3), np.uint8)
cf.img_result = np.zeros((320, 160, 3), np.uint8)

cf.running = True
cf.pause = True
cf.syn = True
cf.time_fps = 0
cf.fps = 0
# Speed Variables
cf.speed = 0

#Angle Variables
cf.angle = 90
cf.center = 160
cf.prev_center = 160
cf.count = 0
cf.error_detect = 0

def main():
    print("Main thread started!")
    time.sleep(1)
    while cf.running:
        if cf.syn:
            cf.count += 1
            if cf.count == 20:
                cf.fps = int(20/(time.time()-cf.time_fps))
                cf.time_fps = time.time()
                cf.count = 0
                print '-----------------fps =', cf.fps
            cf.syn = False
            getLaneCenter(cf.img, True)
            #get_center(cf.img)
            print "center = " + str(cf.center)
            if cf.center > 320 or cf.center < 0:
                continue
            '''
            if abs(cf.center - cf.prev_center) > 20:
                print("???")
                cf.center = cf.prev_center
                cf.error_detect += 1
                if (cf.error_detect == 10):
                    cf.error_detect = 0
                    cf.prev_center = cf.center
            else:
                cf.prev_center = cf.center
            '''
            
                
            cf.angle = PID(cf.center - 160)
            print 100 + cf.angle
            angle_control(100 + cf.angle)
            speed_control(cf.speed)
    #angle_control(90)
    #speed_control(0)
    print("Main thread stopped!")

def show():
    print("Show_thread started !")
    while cf.running:
        cv2.imshow("Lane_find", cf.img_result)
        #cv2.imshow("VIDEO", cf.img)
	
        k = cv2.waitKey(30)
        if k == ord('w'):
            cf.speed += 1
            if cf.speed > 4:
                cf.speed = 4
        if k == ord('s'):
            cf.speed -= 1
            if cf.speed < 0:
                cf.speed = 0

        if k == ord('q'):
            cv2.imwrite('lane.jpg', cf.img)
            cf.speed = 0
            cf.angle = 90

        if k == 27:
            cf.speed = 0
            cf.angle = 90
            cf.running = False
    cv2.destroyAllWindows()
    print("Show thread stopped!")
		
try:
    get_frame_thread = threading.Thread(name = "get_frame_thread", target=get_frame)
    get_frame_thread.start()
    time.sleep(0.5)
    main_thread = threading.Thread(name= "main_thread", target= main)
    main_thread.start()
    show_thread = threading.Thread(name= "show information", target= show)
    show_thread.start()
except Exception as ex:
    print(ex)



# try:
#     cap = cv2.VideoCapture("test2.mp4")
#     while True:
#         _, frame = cap.read()
#         resized = cv2.resize(frame, (480,320), interpolation = cv2.INTER_AREA)
#         cf.img = resized
#         get_center(cf.img)
#         img = cf.img_result.copy()
#         cf.count += 1
#         if cf.count == 20:
#             cf.fps = int(20/(time.time()-cf.time_fps))
#             cf.time_fps = time.time()
#             cf.count = 0
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         #cv2.putText(img,"S: " + str(cf.speed),(90,25), font, 0.8, (255,0,0), 2, 1)
#         #cv2.putText(img,"A: " + str(int(cf.angle)),(350,25), font, 0.8, (255,0,0), 2, 1)
#         cv2.putText(img,"FPS: "+ str(int(cf.fps)),(250,25), font, 1, (255,0,0), 2, 1)

#         cv2.imshow("Result", img)
#         cv2.imshow("Lane_find", cf.img_line)
#         cv2.imshow("VIDEO", cf.img)
#         key = cv2.waitKey(30)
#         if key == 27:
#             break
# except Exception as ex:
#     print(ex)