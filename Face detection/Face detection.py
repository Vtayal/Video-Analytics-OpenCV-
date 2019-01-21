import cv2
import numpy as np

#load the Haarcascade file
face_cascade = cv2.CascadeClassifier("D://UDEMY//Video analytics using OpenCV//Face detection//haarcascade_frontalface_default.xml")

#check if the cascade file ahs been loaded correctly
if face_cascade.empty():
	raise IOError('unable to load the face cascade classifier xml file')

#initialize the video capture object
cap = cv2.VideoCapture(0)

#define the scaling factor
scaling_factor = 0.5

#iterate until the user hits ESC key
while True:
		
	#capture the current frame
	_, frame = cap.read()

	#resize the frame
	frame = cv2.resize(frame, None,fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)

	#convert to grayscale
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#run the face detector on the grayscale image
	face_rects = face_cascade.detectMultiScale(frame_gray,1.3,5)

	#draw a rectangle around the face
	for (x,y,w,h) in face_rects:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

	#display the output
	cv2.imshow('Face detector', frame)

	#check if the user hit the ESC key
	c = cv2.waitKey(1)
	if c == 27:
		break

#release the video capture object
cap.release()

#close all the windows
cv2.destroyAllWindows()