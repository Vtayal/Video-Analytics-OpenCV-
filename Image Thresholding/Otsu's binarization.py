import cv2
import numpy as np
import matplotlib.pyplot as plt

img_file = "D://UDEMY//Video analytics using OpenCV//Image thresholding//sample.jpg"

img = cv2.imread(img_file,0)

#global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#otsu's thresholding after gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plot all the images and their histograms
images = [img,0,th1,
	img,0,th2,
	blur,0,th3]

titles = ['orig noisy img','histogram','global thresholding(v=127)','orig noisy img','histogram',"otsu's thresholding",'gaussian filtered img','histogram',"otsu's thresholding"]

for i in range(3):
	plt.subplot(3,3,i*3+1)
	plt.imshow(images[i*3],'gray')
	plt.title(titles[i*3])
	plt.xticks([]),plt.yticks([])

	plt.subplot(3,3,i*3+2)
	plt.hist(images[i*3].ravel(),256)
	plt.title(titles[i*3+1])
	plt.xticks([]),plt.yticks([])

	plt.subplot(3,3,i*3+3)
	plt.imshow(images[i*3+2],'gray')
	plt.title(titles[i*3+2])
	plt.xticks([]),plt.yticks([])
plt.show()