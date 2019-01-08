import numpy as np
import cv2
import cv2.cv

cap = cv2.VideoCapture(0)

#define the codec and create Video_Writer Object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('D://UDEMY//Video analytics using OpenCV//Video capture//output1.mkv', fourcc, 20.0, (640,480))

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		frame = cv2.flip(frame,0)

		#write the flipped frame
		out.write(frame)

		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	else:
		break

#when everything done, release everything
cap.release()
out.release()
cv2.destroyAllWindows()