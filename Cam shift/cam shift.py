import cv2
import numpy as np

#define a class to handle object tracking related functionality
class ObjectTracker(object):
	def __init__(self):
		
		#initialize the scaling factor
		self.scaling_factor=0.5

		#initialize the video capture object
		self.cap = cv2.VideoCapture(0)
		
		#capture the frame from the webcam
		_,self.frame = self.cap.read()
		
		#scaling factor for the captured frame
		self.frame = cv2.resize(self.frame,None,fx=self.scaling_factor, fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
		
		#create a window to display the frame
		cv2.namedWindow('Object Tracker')
		
		#set the mouse callback function to track the mouse
		cv2.setMouseCallback('Object Tracker', self.mouse_event)
		
		#initialize the variable related to rectangular region selection
		self.selection = None
		
		#initialize the variable related to starting position
		self.drag_start = None
		
		#initialize the variable related to state of tracking
		self.tracking_state = 0

	# define a method to track the mouse events
	def mouse_event(self, event, x, y, flags, param):
		
		#convert x and y coordinates into 16-bit numpy integers
		x, y = np.int16([x, y])

		#check if a mouse button down event has occurred
		if event == cv2.EVENT_LBUTTONDOWN:
			self.drag_start = (x, y)
			self.tracking_state = 0

		#check if the user has started selecting the region
		if self.drag_start:
			if flags & cv2.EVENT_FLAG_LBUTTON:
				
				#extract the dimensions of the frame
				h,w = self.frame.shape[:2]
				
				#get the initial position
				xi,yi = self.drag_start
				
				#get the max and min values
				x0,y0 = np.maximum(0, np.minimum([xi,yi],[x,y]))
				x1,y1 = np.minimum([w,h], np.maximum([xi,yi],[x,y]))
				
				#reset the selection variable
				self.selection = None
				
				#finalize the rectangular selection
				if x1-x0>0 and y1-y0>0:
					self.selection = (x0,y0,x1,y1)
			else:
				#if the selection is done, start tracking
				self.drag_start = None
				if self.selection is not None:
					self.tracking_state = 1

	#method to start tracking the object
	def start_tracking(self):
		#iterate until the user presses the ESC key
		while True:
			
			#capture the frame from webcam
			_, self.frame = self.cap.read()

			
			#resize the input frame
			self.frame = cv2.resize(self.frame, None, fx=self.scaling_factor, fy=self.scaling_factor,interpolation=cv2.INTER_AREA)
			
			#create a copy of the frame
			vis = self.frame.copy()
			
			#convert the frame to HSV colorspace
			hsv = cv2.cvtColor(self.frame,cv2.COLOR_BGR2HSV)
			
			#create the mask based on predefined thresholds
			mask = cv2.inRange(hsv, np.array((0.,60.,32.)),np.array((180.,255.,255.)))
			
			#check if the user has selected the region
			if self.selection:
				
				#extract the coordinates of the selected rectangle
				x0,y0,x1,y1 = self.selection
				
				#extract the tracking window
				self.track_window = (x0,y0,x1-x0,y1-y0)
				
				#extract the regions of interest
				hsv_roi = hsv[y0:y1,x0:x1]
				mask_roi = mask[y0:y1,x0:x1]
				
				#compute the histogram of the region of interest in the HSV image using the mask
				hist = cv2.calcHist([hsv_roi],[0],mask_roi,[16],[0,180])
				
				#normalize and reshape the histogram
				cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
				self.hist = hist.reshape(-1)
				
				#extract the region of interest from the frame
				vis_roi = vis[y0:y1,x0:x1]
				
				#compite the image negative(for display only)
				cv2.bitwise_not(vis_roi, vis_roi)
				vis[mask == 0] = 0

			#check if the system in the 'tracking' mode
			if self.tracking_state == 1:
				
				#reset the selection variable
				self.selection = None
				
				#compute the histogram for the back projection
				hsv_backproj = cv2.calcBackProject([hsv],[0],self.hist,[0,180],1)
				
				#compute bitwise AND between histogram backProjection and the mask
				hsv_backproj &= mask
				
				#define termination criteria for the tracker
				term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
				
				#apply CAMshift on 'hsv_backproj'
				track_box, self.track_window = cv2.CamShift(hsv_backproj, self.track_window, term_crit)
				
				#draw an ellipse around the object
				cv2.ellipse(vis, track_box, (0,255,0), 2)

			#show the output live video
			cv2.imshow('Object Tracker', vis)

			#stop if the user hits the 'ESC key'
			c = cv2.waitKey(5)
			if c == 27:
				break

		#close all the windows
		cv2.destroyAllWindows()

#start tracker
ObjectTracker().start_tracking()