import argparse
import datetime
import imutils
import time
import cv2
 
 
 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=1000, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
	vs = cv2.VideoCapture(0)
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["video"])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
substractor = cv2.bgsegm.createBackgroundSubtractorGMG()

substractor.setBackgroundPrior(0.85)
substractor.setDecisionThreshold(0.65)
substractor.setDefaultLearningRate(0.02)
substractor.setSmoothingRadius(5)

print(' Background prior probability:',substractor.getBackgroundPrior())
print(' Decision threshold:', substractor.getDecisionThreshold())
print(' Learning rate:', substractor.getDefaultLearningRate())
print(' Smoothing radius:', substractor.getSmoothingRadius())

while True:
    
	read, frame = vs.read()
	frame = imutils.resize(frame, width=1000)
	# gray = cv2.GaussianBlur(frame, (21, 21), 0)

	mask = substractor.apply(frame)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	dilated_mask = cv2.dilate(mask, None, iterations=2)
 
	cnts = cv2.findContours(dilated_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	for c in cnts:
		if cv2.contourArea(c) < args["min_area"]:
			continue

		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
  
	cv2.imshow("Dilated mask", dilated_mask)
	cv2.imshow("Mask", mask)
	cv2.imshow("Security Feed", frame)
	cv2.waitKey(2)
 
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()