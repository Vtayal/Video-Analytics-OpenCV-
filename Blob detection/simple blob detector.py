import cv2
import cv2.cv as cv
import numpy as np
import imageio

vid = imageio.get_reader("sample.mp4",'ffmpeg')
#fourcc = cv2.cv.FOURCC(*"DIB")
#video = cv2.VideoWriter("bgsub.mp4",fourcc,30(900,1200))
o = imageio.get_writer("bgsub.mp4")
final = imageio.get_writer('count.mp4')
fgbg