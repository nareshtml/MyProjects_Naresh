import cv2
import os
import numpy as np

def detect_face(img):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
	face_detect=cv2.CascadeClassifier("lbpcascade_frontalface.xml");
	faces=face_detect.detectMultiScale(gray,scalefactor=1.2,minNeighbors=5)
	if(len(faces)==0):
		return None,None;
	(x,y,w,h)=faces[0]
	return gray[y:y+w,x:x+h],faces[0]

def prepare_training(data):
	dirs=os.listdir(data)
	faces=[]
	labels=[]
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue
		label=int(dir_name,replace("s",""))
		subject_dir=data+"/"+dir_name
		sub_imgname=os.listdir(subject_dir)
		for img_name in sub_imgname:
			if img_name.startswith("."):
				continue
			image_path=subject_dir+"/"+img_name
			img=cv2.imread(image_path)
			cv2.imshow("Training>>",img)
			cv2.waitKey(100)
			
			face,rect=detect_face(img)
			if face is not None:
				faces.append(face)
				labels.append(label)
	
	cv2.waitKey(1)
	cv2.destroyAllWindows()
	return faces,labels

print("Preparing data..")
faces,labels=prepare_training("training")
print ("data prepared")
print("Total faces:",len(faces))
print("Total labels:",len(labels))

face_recognize=cv2.face.LBPHFaceRecognizer_create()
face_recognize.train(faces,np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img) 
    label= face_recognizer.predict(face)
    label_text = subjects[label[0]]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img		
print("Predicting images...")

test_img1 = cv2.imread("test-data/1.jpg")
test_img2 = cv2.imread("test-data/2.jpg")


predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

cv2.imshow(subjects[1], predicted_img1)
cv2.imshow(subjects[2], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()	
		
		
		
		
		
		
		
		
			
