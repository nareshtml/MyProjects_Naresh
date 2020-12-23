import cv2

p_src='pedestrian.xml'
vid_src='pedestrian.avi'
cam=cv2.VideoCapture(vid_src)
cascade=cv2.CascadeClassifier(p_src)

while True:
	ret,img=cam.read()
	if(type(img)==type(None)):
		break
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	detect=cascade.detectMultiScale(gray,1.3,2)

	for (x,y,w,h) in detect:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
	cv2.imshow('video',img)
	if cv2.waitKey(23)==30:
		break
cv2.destroyAllWindows()