import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("D://UDEMY//Video analytics using OpenCV//Image loading program//sample.jpg")

blur = cv2.blur(img,(5,5))
gaussian = cv2.GaussianBlur(img,(5,5),0)
median = cv2.medianBlur(img,5)

plt.subplot(221)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])
#plt.show()

plt.subplot(222)
plt.imshow(blur)
plt.title('Blurred')
plt.xticks([])
plt.yticks([])
#plt.show()

plt.subplot(223)
plt.imshow(gaussian)
plt.title('Gaussian filtered')
plt.xticks([])
plt.yticks([])
#plt.show()

plt.subplot(224)
plt.imshow(median)
plt.title('Median filtered')
plt.xticks([])
plt.yticks([])
plt.show()