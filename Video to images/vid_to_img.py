import cv2
vidcap = cv2.VideoCapture('D://experimenting-with-sort//ppp.avi')
success,image = vidcap.read()
print ("success")
count = 0
while success:
	cv2.imwrite("D://experimenting-with-sort//test//Pictures%d.jpg" % count, image) # save frame as JPEG file
	success,image = vidcap.read()
	print('Read a new frame: ', success)
	count += 1