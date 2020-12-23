import cv2

print("Vehicle detection...")

cascade_src='two_wheeler.xml'
vid_src='two_wheeler.mp4'
cam=cv2.VideoCapture(vid_src)
cascade=cv2.CascadeClassifier(cascade_src)

while True:
	ret,img=cam.read()
	if(type(img)==type(None)):
		break
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cars=cascade.detectMultiScale(gray,2.3,2)
	for(x,y,w,h) in cars:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,215),2)
	cv2.imshow('video',img)
	if cv2.waitKey(23)==27:
		break
cv2.destroyAllWindows()	