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

def drawMatches(img1, kp1, img2, kp2, matches, img3, h, w, n_x1, n_y1, n_x2, n_y2):
    """
    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are traced with circles, while lines are connected
    between matching keypoints.

    img1 - Reference image template(Grayscale images)
	img2 - Trained image data(Grayscale images)
	n_x1, n_y1, n_x2, n_y2 - Coordinates of the region of interest (based on Detection)
	img3 - Image Output after image detection and recognition analysis performed
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    tup1 = []
    tup2 = []
    
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        print "\n\nkeypoint->img1_idx :", img1_idx
        print "keypoint->img2_idx :", img2_idx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        print "(x1,y1) :", (x1,y1)
        print "(x2,y2) :", (x2,y2)  

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        tup1.append((x1,y1))
        tuple(tup1)
        tup2.append((x2,y2))
        tuple(tup2)
        # end for loop
        

    sizeTuple1 = len(tup1)
    sizeTuple2 = len(tup2)

    #print "\n\nsize tuple 1 :", sizeTuple1
    #print "size tuple 2 :", sizeTuple2
    print "\n\n"

    print "10 lines Detection -> Each line is weighed by 10%. Thus 10 lines = 100%\n"
    

    print "\n"
    counterT = 0 # Flag counter
    countNow = 0 # Count each detected feature
    for (a,b) in tup1:
        (p,q) = tup2[counterT]
        if (counterT < len(tup2)):
            if (a,b) != (p,q):
                print (a,b), " compare with ", (p,q)
                #print "\nFalse"
                if ((p-a) >= 112 and (p-a) <= 119) and ((q-b) >= 105 and (q-b) <= 110):
                    countNow += 1
                if ((p-a) >= 87 and (p-a) <= 90) and ((q-b) >= 94 and (q-b) <= 98):
                    countNow += 1
                if ((p-a) >= 30 and (p-a) <= 73) and ((q-b) >= 49 and (q-b) <= 62):
                    countNow += 1
                if ((p-a) >= 5 and (p-a) <= 77) and ((q-b) >= -25 and (q-b) <= -2):
                    countNow += 1                    
            else:
                #print "\nTrue"
                print "\nOther case"
        counterT += 1
        #print "counter: ", counterT
        print "\n"

    #print "\nTotal count:", countNow
    print "\nNumber of matched features are :", countNow, " out of 10"
    print countNow*10, "% scores of image accuracy "

    print "\n"
	
	'''
	This section shows that all the detected features from (trained) data image based on keypoints are checked/located within
	a region of interest (detection area)
	'''
    newCount = 0
    featuresCount = 0
    print n_x1, n_y1, n_x2, n_y2
    for (i,j) in tup2:
        if (i >= n_x1 and i <= n_x2) and (j >= n_y1 and j <= n_y2):
            print "True. Detected features ARE within a region of interest"
            featuresCount += 1 
        else:
            print "False. Detected features ARE NOT within a region of interest"
    print "\nNumber of matched features are :", featuresCount, " out of 10"
    print featuresCount*10, "% scores of image accuracy "
    

	
	'''
	This section shows the final output for recognition based on the percentage of scores from
	the similarity of image pattern between two images
	'''
    # Write/Display some Text
    font = cv2.FONT_HERSHEY_SIMPLEX

    #matchedCount = countNow*10
    matchedCount = featuresCount*10
    if (matchedCount >= 80):
        print "\nThe sign HAS SAME IMAGE PATTERN with the template"
        
        cv2.putText(img3,'Scores:'+ str(matchedCount) + '%',(w/5, h-70), font, 0.7,(0,0,255),2)
        cv2.putText(img3,'Sign Recognised',(w/5, h-50), font, 0.7,(0,0,255),2)
    if (matchedCount > 50 and matchedCount < 80):
        print "\nThe sign HAS FEW SIMILARITY IN IMAGE PATTERN"
        
        cv2.putText(img3,'Scores:'+ str(matchedCount) + '%',(w/5, h-70), font, 0.7,(0,0,255),2)
        cv2.putText(img3,'Almost Recognised',(w/5, h-50), font, 0.7,(0,0,255),2)
    if (matchedCount <= 50):
        print "\nThe sign HAS NO SIMILAR PATTERN with the template"

        cv2.putText(img3,'Scores:'+ str(matchedCount) + '%',(w/5, h-70), font, 0.7,(0,0,255),2)
        cv2.putText(img3,'Sign Not Matched',(w/5, h-50), font, 0.7,(0,0,255),2)

    

    # Show/display the output of detected image
    print "\n"
    cv2.imshow('Recognition Result', img3)
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')
    cv2.destroyAllWindows()
    #cv2.destroyAllWindows('Matched Features')

    # Return the image 
    return out
