'''
# This code is to perform detection and recognition analysis
# Programmer: Muhammad Hafidz Misrudin, N8448141
# Method of implementation: Feature Matching using SIFT/Orb descriptors
# It requires an OPENCV library and additional (image processing) packages in order to perform the tasks
# OPENCV versions: 2.4.11 or higher and 3.0 (preferred) 

# Programmer: Muhammad Hafidz Misrudin, N8448141
# Mostly the code has been consulted from OPENCV documentations.
# However, some of it has been improved and improvised throughout development.
# Therefore, it is entitled for copyright protection.

# Last developed/tested (analysis): Friday 23rd October 2015
# Last updated: Monday 2nd November 2015
'''

import numpy as np
import cv2
from drawMatches import *


img1 = cv2.imread('/home/muhammad/Desktop/Final/BF_SIFT/crop/ref_temp/type12/t12_1.jpg', 0) # Cropped image - ensure grayscale
img2 = cv2.imread('/home/muhammad/Desktop/Final/BF_SIFT/img/type12/t12_1.jpg', 0)  # Original image - ensure grayscale
img3 = cv2.imread('/home/muhammad/Desktop/Final/BF_SIFT/img/type12/t12_1.jpg')
height, weight, channels = img3.shape

# Need to be manually changed (hardcode)
color = 'W' # Colors: R, B, Y, G, W

hsv_img = cv2.cvtColor(img3, cv2.COLOR_BGR2HSV)

'''
The collections of arrays according to respective HSV colours
For example, Set an array for lower resolution and set an array for higher resolution for threshold purposes
'''
lower_red = np.array([170,150,50],dtype=np.uint8)
upper_red = np.array([179,255,255],dtype=np.uint8)

lower_blue = np.array([100,100,100], dtype=np.uint8)
upper_blue = np.array([120,255,255], dtype=np.uint8)

lower_yellow = np.array([20, 80, 80], dtype=np.uint8)
upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

lower_green = np.array([60, 55, 0], dtype=np.uint8)
upper_green = np.array([100, 255, 120], dtype=np.uint8)

sensitivity = 30
#sensitivity = 50
lower_white = np.array([0,0,255-sensitivity], dtype=np.uint8)
upper_white = np.array([255,sensitivity,255], dtype=np.uint8)


if (color == 'R'):
    lower_c = lower_red
    upper_c = upper_red
if (color == 'B'):
    lower_c = lower_blue
    upper_c = upper_blue
if (color == 'Y'):
    lower_c = lower_yellow
    upper_c = upper_yellow
if (color == 'G'):
    lower_c = lower_green
    upper_c = upper_green
if (color == 'W'):
    lower_c = lower_white
    upper_c = upper_white
    

frame_threshed = cv2.inRange(hsv_img, lower_c, upper_c)
imgray = frame_threshed
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt = contours[max_index]

######################################## Detection Analysis Section ################################################
####################################################################################################################

x,y,w,h = cv2.boundingRect(cnt)
print x,y,w,h
x1 = x
y1 = y
x2 = x+w
y2 = y+h
 
print "(X1,Y1)", (x1, y1)
print "(X2,Y2)",(x2, y2)

cv2.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),3)
cv2.imshow("Image Detection Result",img3)

####################################################################################################################
######################################## Detection Analysis Section ################################################


######################################## Recognition Analysis Section ################################################
####################################################################################################################


# Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
orb = cv2.ORB(1000, 1.2)
#orb = cv2.ORB()
#sift = cv2.SIFT(1000, 1.2)


# Alternative: Detect keypoints of original image (using SIFT object detector)
#(kp1, des1) = sift.detectAndCompute(img1,None)
#(kp2, des2) = sift.detectAndCompute(img2,None)

# Detect keypoints of original image (using orb object detector)
(kp1,des1) = orb.detectAndCompute(img1, None)


# Detect keypoints of cropped image
(kp2,des2) = orb.detectAndCompute(img2, None)


# Create (Brute Force) matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


# Do matching
matches = bf.match(des1,des2)
#print matches


# Sort the matches based on distance.  The Least distance is better
matches = sorted(matches, key=lambda val: val.distance)

# Show only the top 10 (line) matches
out = drawMatches(img1, kp1, img2, kp2, matches[:10], img3, height, weight, x1, y1, x2, y2)
#out = drawMatches(img1, kp1, img2, kp2, matches[:10])


##cv2.waitKey(0)
##cv2.destroyWindow()

####################################################################################################################
######################################## Recognition Analysis Section ################################################
