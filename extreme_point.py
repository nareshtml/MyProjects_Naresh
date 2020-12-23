import imutils
import cv2

ima=cv2.imread("extreme.jpg");
gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY);
gray=cv2.GaussianBlur(gray,(5,5),0);

thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

cv2.drawContours(ima, [c], -1, (0, 255, 255), 2)
cv2.circle(ima, extLeft, 8, (0, 0, 255), -1)
cv2.circle(ima, extRight, 8, (0, 255, 0), -1)
cv2.circle(ima, extTop, 8, (255, 0, 0), -1)
cv2.circle(ima, extBot, 8, (255, 255, 0), -1)

cv2.imshow("Image", ima)
cv2.waitKey(0)