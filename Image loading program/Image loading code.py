#A simple image loading program where an image is shown 3 ways
#Original, alpha channel and gray

import cv2
import numpy as np

img_file = "sample.jpg"

img = cv2.imread(img_file, cv2.IMREAD_COLOR) 			#rgb
alpha_img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED) 		#rgba
gray_img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) 		#grayscale

print(type(img),'\n')
print('RGB shape-',img.shape,'\n') 			#rows, columns, channels
print('ARGB shape-',alpha_img.shape,'\n')		#rows, columns, channels
print('GRAY shape-',gray_img.shape,'\n')		#rows, columns
print('img datatype-',img.dtype,'\n')
print('img size-',img.size,'\n')