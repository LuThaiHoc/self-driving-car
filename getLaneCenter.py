import cv2
import numpy as np
import config as cf

def getClosing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 60, 5)
    # cv2.imshow('edges', edges)
    kernel = np.ones((9,9),np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closing

def getDilate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 70, 5)
    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(edges, kernel, iterations=1)
    return dilation

def getLaneCenter(image):
    im = np.copy(image)
    #closing = getClosing(image)
    dilation = getDilate(image)
    #cv2.imshow('closing', closing)
    #cv2.imshow('dilation', dilation)
    #mask = cv2.bitwise_not(closing, closing)
    mask = cv2.bitwise_not(dilation, dilation)
    #cv2.imshow('massk', mask)
    #mask[dilation > 100] = 0
    # cv2.imshow('canny value', mask)

    #final = np.zeros(mask.shape)
    #im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1] # get largest five contour
    cX = 160
    cY = 120
    if (len(contours) != 0):
        maxCnt = max(contours, key = cv2.contourArea)
        # print ('area= ',cv2.contourArea(maxCnt)
        if (cv2.contourArea(maxCnt) > 12000):
            M = cv2.moments(maxCnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print (cX, cY)
            #if (abs(cY - 100) > 30):
             #   cX = 160
              #  cY = 120
            cv2.drawContours(im, [maxCnt], 0, (0,255,0), -1)
            cv2.circle(im, (cX, cY), 3, (0,0,255), -1)
                #cv2.drawContours(final, contours, 0, 255, -1)

    # cv2.imshow('final', final)
    #kernel = np.ones((5,5),np.uint8)
    #dilation = cv2.dilate(final, kernel, iterations=1)
    cf.center = cX
    cf.img_result = im
    return im, (cX, cY)

def getLaneCenter2(image):
    #im = np.copy(image)
    #closing = getClosing(image)
    dilation = getDilate(image)
    #cv2.imshow('closing', closing)
    #cv2.imshow('dilation', dilation)
    #mask = cv2.bitwise_not(closing, closing)
    mask = cv2.bitwise_not(dilation, dilation)
    #cv2.imshow('massk', mask)
    #mask[dilation > 100] = 0
    # cv2.imshow('canny value', mask)

    #final = np.zeros(mask.shape)
    #im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    _, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1] # get largest five contour
    cX = 160
    cY = 120
    if (len(contours) != 0):
        maxCnt = max(contours, key = cv2.contourArea)
        # print ('area= ',cv2.contourArea(maxCnt)
        if (cv2.contourArea(maxCnt) > 12000):
            M = cv2.moments(maxCnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print (cX, cY)
            #if (abs(cY - 100) > 30):
             #   cX = 160
              #  cY = 120
            #cv2.drawContours(im, [maxCnt], 0, (0,255,0), -1)
            #cv2.circle(im, (cX, cY), 3, (0,0,255), -1)
                #cv2.drawContours(final, contours, 0, 255, -1)

    # cv2.imshow('final', final)
    #kernel = np.ones((5,5),np.uint8)
    #dilation = cv2.dilate(final, kernel, iterations=1)
    #cf.center = cX
    return cX

def testVideo():
    cap = cv2.VideoCapture('Data/video/public2.avi')
    #cap = cv2.VideoCapture(0)
    while(True):
        _, frame = cap.read()
        frame = cv2.resize(frame, (320,240))
        #img = frame[80:240,:]
        # image = np.copy(frame)
        getLaneCenter(frame)
        #im, (cX, cY) = getLaneCenter(frame)
        #print (cX, cY)
        #cv2.imshow('Detect', im)
        Key = cv2.waitKey(1)
        if (Key == 27):
            break
        if (Key == ord('s')):
            cv2.waitKey(0)


#testVideo()


