import cv2;

def getmysketch(img):
	grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
	gaussianImg=cv2.GaussianBlur(grayImage,(3,3),0);
	edgeImg=cv2.Canny(gaussianImg,100,200);
	a,threshImg=cv2.threshold(edgeImg,120,2255,cv2.THRESH_BINARY_INV);
	return threshImg;

vid_cap=cv2.VideoCapture(0);

while True:
	ret,pic=vid_cap.read();
	cv2.imshow("your sketch is ",getmysketch(pic));
	if cv2.waitKey(1)==12:
		break;
vid_cap.release();
cv2.destroyAllWindows();
	