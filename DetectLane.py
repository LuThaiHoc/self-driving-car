import cv2
import numpy as np
import glob
import config as cf
import math
import time


'''
video source: video/video.avi
images source: images/*.jpg
shadow, snow cut from image (for detect color of shadow, snow, v.v..): cut/*.jpg

'''

# m is an chanel splitted from HSV image
def getAvgAndVariance(m):
    '''
        average = (m1 + m2 + ... + mN) / N
        variance = sum(m(i) - average)^2 / N
    '''
    avg = m.mean()
    temp = np.square(np.subtract(m, avg))
    variance = np.sum(temp) / (temp.shape[0]*temp.shape[1])
    #variance = np.sqrt(variance)
    return avg, variance

#get h_average, h_variance, s_average, s_variance of each lane color
#all lane image file get from folder /lane/*.jpg
files = None
def getColor():
    img = cv2.imread('lane0.jpg')
    blur = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    avg_v = v.mean()
    h_avg, h_variance = getAvgAndVariance(h)
    s_avg, s_variance = getAvgAndVariance(s)
    v_avg, v_variance = getAvgAndVariance(v)
    if (h_variance == 0):
        h_variance = 0.000000001
    color = (h_avg, h_variance, s_avg, s_variance, avg_v)
    print 'color', color
    return color



# m is an chanel splitted from HSV image
#return matrix likelihood of all pixel in chanels
def getChanelLikelihood(m, avg, variance):
    '''
    likelihood = exp(-sqr(m-m_avg) / sqr(m_variance))
    '''
    temp = np.square(m - avg)
    temp = temp / np.square(variance)
    temp = np.exp(-temp)
    return temp

#get likelihood matrix of each chanel
def getLikelihood(hsv_image, lane_color):
    h_avg, h_variance, s_avg, s_variance, avg_v = lane_color

    h_chanel , s_chanel, v_chanel = cv2.split(hsv_image)
    
    h_likelihood = getChanelLikelihood(h_chanel, h_avg, h_variance)
    s_likelihood = getChanelLikelihood(s_chanel, s_avg, s_variance)
    
    likelihood = (h_likelihood, s_likelihood, avg_v)
    return likelihood

#get mask of lane by likelihood
def getMask(image, color):
    blur_img = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
    hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)
    
    # h_chanel , s_chanel, v_chanel = cv2.split(hsv_img)
    _ , _, v_chanel = cv2.split(hsv_img)
    h_likelihood, s_likelihood, avg_v = getLikelihood(hsv_img, color)
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[h_likelihood*s_likelihood > 0.01] = 255
    mask2 = np.zeros((image.shape[0], image.shape[1]))
    mask2[abs(v_chanel - avg_v) < 70] = 255
    mask = cv2.bitwise_and(mask, mask2)
    #mask = preProcess(mask, minsize=5000)
    #mask = np.uint8(mask)
    return mask


def preProcess(mask, minsize):
    mask = np.uint8(mask)
    black = np.zeros(mask.shape,np.uint8)
    
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    _, contours,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contours,_ = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        #print(area)
        if area > minsize:
            approx = cv2.approxPolyDP(cnt,2,True)
            cv2.drawContours(black, [approx], -1, 255, -1)
    return black



color = getColor()

def birdViewTransform(image):
        # obtain a consistent order of the points and unpack them
        # individually
    rect = np.array([
                    [40, 15],
                    [280, 15],
                    [600, 80],
                    [-180, 80]], dtype="float32")
    tl, tr, br, bl = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    # image = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    warped = cv2.resize(warped, (200, 150))
    # return the warped image
    return warped

v = 15
#r = 3600/d (d = 0->60)
def Score(img,side, v, d, drawDirect = False):
  #img : 2D grayScale size (200x150)
  #v : Van toc
  #r : ban kinh cung tron quy dao xe
  #d : goc cua xe
  #bat dau tu chinh giua phia duoi cua anh (vi tri xe hien tai)
  if (abs(d) < 1):
    v = 10
  
  r = 1e10
  if (float(d) != 0):
    r = 3600 / (float)(d)

  #print 'r = ', r
  
  x_org = img.shape[1]/2
  y_org = img.shape[0] - 1

  Max = 0
  x_Max = 100

  t = 0
  x = x_org
  y = y_org

  x_last = x
  y_last = y

  Max = 0

  while(True):
    if drawDirect:
      cv2.circle(img, ((int)(x),(int)(y)), 1, 0 , 2)

    x_last = x
    y_last = y  

    t += 3
    #t += 0.5
    x = int (x_org + r*(1 - math.cos((float)(t)*(v/r))))
    y = int(y_org - r*math.sin((float)(t)*(v/r)))

    #Di vuot ra bien cua anh
    if (x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]):
      break
    #tu vung sang di vao vung toi
    try:
      if (img[y,x] <= 100 and img[y_last,x_last] > 100):
        break
    except Exception:
        break

    score = 0
    if (side == None): #straight
      # score = 150-y - abs(100-x)/2
      score = 150-y - abs(100-x)/2
    elif (side == 0): #left
      score = (150-y)/8 + 2*(100-x)
      if (x > 100): #khong duoc re phai khi co bien bao trai
        score = -1
    elif (side == 1): 
      score = (150-y)/8 + 2*(x-100)
      if (x < 100): #khong duoc re trai khi co bien bao phai
        score = -1

    if (Max <  score and img[y,x] > 100):
        Max = score

  return Max

#img is binary image
def getAngle(img, side):

  # out = np.copy(img)

  # out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
  # out[img > 100] = (255,255,255)
  
  # outtt = np.copy(out)

  Scores = []

  for d in range(5,80,20):
    Max =  Score(img,side, v, d)
    Scores.append((Max, d))
    Max =  Score(img,side, v, -d)
    Scores.append((Max, -d))
  Max =  Score(img,side, v, 1e-10)
  Scores.append((Max, 1e-10))

  for d in range(1, 21, 5):
    Max =  Score(img,side, v, d)
    Scores.append((Max, d))
    Max =  Score(img,side, v, -d)
    Scores.append((Max, -d))

  Scores.sort(reverse = True)
  top_3 = [Scores[i][1] for i in range(0,3,1)]

  score = None
  #print ('top3', top_3)
  avg = np.mean(top_3, axis=0)
  # score =  Score(img,side, v, avg)
  if (side != None and Max == 0):
    if (side == 0):
      avg = -50
    else:
      avg = 50
    # score = Score(img,side, v, avg)
  if (side == None and Max == 0):
    avg = 0
  score =  Score(img,side, v, avg, True)

  return avg, score

def get_center(img):
    # print ('Get angle...')
    # merge = getMergeMask(img)
    im = np.copy(img)
    mask = getMask(img, color)
    #cf.detect = np.copy(img)
    #cf.detect[merge > 100, 1] = cf.detect[merge > 100, 1] + 100
    
    '''
    trans = merge[80:240,:]
    birdview = birdViewTransform(trans)
    avg, score = getAngle(birdview, None)
    cv2.imshow('birdview', birdview)
    cv2.waitKey(10)
    '''
    cv2.imshow('mask', mask)
    mask = np.uint8(mask)
    im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
 
    cX = 160
    cY = 120
    if (len(contours) != 0):
        maxCnt = max(contours, key = cv2.contourArea)
        # print ('area= ',cv2.contourArea(maxCnt)
        if (cv2.contourArea(maxCnt) > 12000):
            M = cv2.moments(maxCnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print (cX, cY)
    
            cv2.drawContours(im, [maxCnt], 0, (0,255,0), -1)
            cv2.circle(im, (cX, cY), 3, (0,0,255), -1)
    #cv2.imshow('Maskkk', mask)
    #cv2.imshow('img', im)


    cf.center = cX
    cf.img_result = im
    return cX, im

    # print('score', avg)
    # cv2.imshow('Direct', out)
    # cv2.imshow('my', cf.birdview)
    # cv2.imshow('video', cf.detect)
'''    
img = cv2.imread('lane.jpg')
img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)

x, im = get_center(img)
print('result', x)
cv2.imshow('image', im)
cv2.waitKey(0)
'''