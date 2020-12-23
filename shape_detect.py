import cv2
import imutils
class shape_detect:
	def __init__(self):
		pass
	def detect(self,c):
		shape="unidentified"
		perimeter=cv2.arcLength(c,True)
		approx=cv2.approxPolyDP(c,0.04*perimeter,True)
		if len(approx==3):
			shape="Triangle"
		elif len(approx)==4:
			(x,y,w,h)=cv2.boundingRect(approx)
			av=w/float(h)
			shape="Square " if av>=0.95 and av<=1.05 else "Rectangle"
		elif len(approx)==5:
			shape="Pentagon"	
		else:
			shape="Circle"
		return shape

img = cv2.imread('shape.jpg')
resized = imutils.resize(img, width=300)
ratio = img.shape[0] / float(resized.shape[0])

gray = cv2.cvtColor(resized , cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray , (5 , 5) , 0)
thresh = cv2.threshold(blurred , 60 , 255 , cv2.THRESH_BINARY)[1]

cnts = cv2.findContours(thresh.copy() , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = shape_detect()

for c in cnts:
    M = cv2.moments(c)

    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)
    c = c.astype("float")
    
    c *= ratio
    
    c = c.astype("int")
    
    cv2.drawContours(img , [c] , -1 , (0 , 255 , 0) , 2)
    
    cv2.putText(img , shape , (cX , cY) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (255 , 255 , 255) , 2)
    
    cv2.waitKey(0)