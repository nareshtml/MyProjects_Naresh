import numpy as np
import cv2 as cv

# load the image using cv2.imread function in grayscale
img = cv.imread("featur.jpg",cv.IMREAD_GRAYSCALE)

#create ORB object
orb = cv.ORB_create(nfeatures=1500)

# detecting the feature
keypoints, descriptors = orb.detectAndCompute(img, None)

# it will draw the Keypoints that detect using ORB
img = cv.drawKeypoints(img, keypoints, None)

#show the image as output
cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()