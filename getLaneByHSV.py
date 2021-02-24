import cv2
import numpy as np
import config as cf



average = [ 69,  48, 150] #giay A4


r = (124, 166, 55, 38) #defaut rect to crop
cX = 160    #center
cY = 120
def getLaneCenter(img, getAVG = False):
    global average, cX, cY
    im = np.copy(img)
    blur_img = cv2.medianBlur(img, 3)
    #cv2.imshow('Blur', blur_img)

    hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    #cv2.imshow('hsv', hsv)  
   
    # Select ROI
    #print(cX, cY)
    if (getAVG):
        #r = (124, 166, 55, 38)
        #roi = hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        roi = hsv[(cY-10):(cY+10) , (cX - 10):(cX+10)]
        average = roi.mean(axis=0).mean(axis=0)
        #print 'average', average
    #r = cv2.selectROI(img)
    #print (r)
    

    # define range of color in ROI (using HSV color chanel)
    lower_road = np.array([average[0]-20,average[1]-20,average[2]-30])
    upper_road = np.array([average[0]+20,average[1]+20,average[2]+30])
    #lower_road = np.array([average[0]-20,average[1]-20,average[2]-25])
    #upper_road = np.array([average[0]+20,average[1]+20,average[2]+25])

    # Threshold the HSV image to get only road colors
    mask = cv2.inRange(hsv, lower_road, upper_road)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
 
    cX = 160
    cY = 120
    if (len(contours) != 0):
        maxCnt = max(contours, key = cv2.contourArea)
        #print('area= ',cv2.contourArea(maxCnt))
        if (cv2.contourArea(maxCnt) > 10000):
            M = cv2.moments(maxCnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #print (cX, cY)
            
            cv2.drawContours(im, [maxCnt], 0, (0,255,0), -1)
            cv2.rectangle(im, (cX-10, cY-10), (cX+10, cY+10), (255,0,0), 2)
            cv2.circle(im, (cX, cY), 3, (0,0,255), -1)
    #cv2.imshow('Maskkk', mask)
    #cv2.imshow('img', im)


    cf.center = cX
    cf.img_result = im
    return cX, im


def test():
    img = cv2.imread('lane.jpg')
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)

    x, im = getLaneCenter(img, True)
    print('result', x)
    cv2.imshow('image', im)
    cv2.waitKey(0)
    
#test()

