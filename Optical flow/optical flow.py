import cv2
import numpy as np

#define a function to track the object
def start_tracking():

	#initialize the video capture object
	cap = cv2.VideoCapture(0)

	#define the scaling factor for the frames
	scaling_factor = 0.5

	#number of frames to track
	num_frames_to_track = 5

	#skipping factor
	num_frames_jump = 2

	#initialize variables
	tracking_paths = []
	frame_index = 0

	#define tracking parameters
	tracking_params = dict(winSize=(11,11),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,0.03))

	#iterate until the user hits ESC key
	while True:
		
		#capture the current frame
		_, frame = cap.read()

		#resize the frame
		frame = cv2.resize(frame, None,fx=scaling_factor, fy=scaling_factor,interpolation=cv2.INTER_AREA)

		#convert to grayscale
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#create a copy of the frames
		output_img = frame.copy()

		if len(tracking_paths)>0:
			
			#get images
			prev_img, current_img = prev_gray, frame_gray

			#organize the feature points
			feature_points_0 = np.float32([tp[-1] for tp in tracking_paths]).reshape(-1,1,2)

			#compute optical flow
			feature_points_1, _, _ = cv2.calcOpticalFlowPyrLK(prev_img, current_img, feature_points_0, None, **tracking_params)

			#compute reverse optical flow
			feature_points_0_rev, _, _ = cv2.calcOpticalFlowPyrLK( current_img, prev_img, feature_points_1, None, **tracking_params)

			#compute the differences between forward and reverse optical flow
			diff_feature_points = abs(feature_points_0 - feature_points_0_rev).reshape(-1,2).max(-1)

			#extract the good points
			good_points = diff_feature_points<1

			#initialize variable
			new_tracking_paths = []

			#iterate through all the good features points
			for tp, (x,y), good_points_flag in zip(tracking_paths, feature_points_1.reshape(-1,2), good_points):
				
				#if the flag is not true, then continue
				if not good_points_flag:
					continue

				#append X and Y coordinates and check if its length greater than the threshold
				tp.append((x,y))
				if len(tp) > num_frames_to_track:
					del tp[0]

				new_tracking_paths.append(tp)

				#draw a circle around the feature points
				cv2.circle(output_img,(x,y),3,(0,255,0),-1)

			#update the tracking paths
			tracking_paths = new_tracking_paths

			#draw lines
			cv2.polylines(output_img, [np.int32(tp) for tp in tracking_paths], False, (0,150,0))

		#go into this 'if' condition after skipping the right number of frames
		if not frame_index %  num_frames_jump:
			
			#create a mask and draw the circles
			mask = np.zeros_like(frame_gray)
			mask[:] = 255
			for x,y in [np.int32(tp[-1]) for tp in tracking_paths]:
				cv2.circle(mask, (x,y), 6, 0, -1)

			#compute good features to track
			feature_points = cv2.goodFeaturesToTrack(frame_gray, mask = mask, maxCorners=500, qualityLevel=0.3,minDistance=7, blockSize=7)

			#check if feature point exist. if so, then append them to the tracking paths
			if feature_points is not None:
				for x,y in np.float32(feature_points).reshape(-1,2):
					tracking_paths.append([(x,y)])

		#update variables
		frame_index += 1
		prev_gray = frame_gray

		#display output
		cv2.imshow('optical flow', output_img)

		#check if the user hit the ESC key
		c = cv2.waitKey(1)
		if c == 27:
			break

#start the tracker
start_tracking()

#close all the windows
cv2.destroyAllWindows()