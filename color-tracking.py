# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
from scipy.stats import itemfreq
import math as math


cap = cv2.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window

# mouse callback function
img = np.zeros(frame.shape, np.uint8)
drawing = False # true if mouse is pressed
wait = True
x0,y0 = -1,-1
roi_hist = []

def init_track_window(x0,y0,x1,y1):
    global track_window,roi_hist,greenLower,greenUpper
    c,r,w,h= min(x0,x1),min(y0,y1),abs(x1-x0),abs(y1-y0)  # simply hardcoded the values

    track_window = (c,r,w,h)
    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w]

    roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    average_color = [roi[:, :, i].mean() for i in range(roi.shape[-1])]
    arr = np.float32(roi)
    pixels = arr.reshape((-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(roi.shape)
    # print(quantized[:,:,0])
    hue, count = np.unique(quantized[:,:,0],False,False,True)
    fcolor = hue[np.argmax(count)]
    print(palette)

    s = 0
    v = 0
    c = 0
    for x in palette:
        if x[0] == fcolor :
            s += x[1]
            v += x[2]
            c += 1

    avg_s = math.floor(s / c)
    avg_v = math.floor(v / c)
    top_hue = fcolor + 5
    bottom_hue = fcolor - 5
    top_s = avg_s + 40
    bottom_s = avg_s - 50
    top_v = avg_v + 40   
    bottom_v = avg_v - 80
    if top_hue > 255 :
      top_hue = 255
    if bottom_hue < 0 :
      bottom_hue = 0
    if top_s > 255 :
      top_s = 255
    if bottom_s < 0 :
      bottom_s = 0
    if top_v > 255 :
      top_v = 255
    if bottom_v < 0 :
      bottom_v = 0
    print top_hue
    print bottom_hue
    print top_s
    print bottom_s
    # top_v=255
    # bottom_v=0
    greenLower = (bottom_hue, bottom_s, bottom_v)
    greenUpper = (top_hue, 255, top_v)    
    print [fcolor,avg_s,avg_v]
    # print(frec[np.argmax(frec[:, -1])])
    # print(dominant_color)
    # cv2.imshow('color',quantized)
# 173r 62g 67b
# [4, 219.0, 124.0]
# [2, 255.0, 113.0]

    # exit()
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0, 60,32)), np.array((180,255,255)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)


# mouse callback function
def draw_rect(event,x,y,flags,param):
    global x0,y0,drawing,mode,wait,img
    if event == cv2.EVENT_LBUTTONDOWN:
        img = np.zeros(frame.shape, np.uint8)
        drawing = True
        x0,y0 = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = np.zeros(frame.shape, np.uint8)
            cv2.rectangle(img,(x0,y0),(x,y),(0,255,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        wait = False
        drawing = False
        init_track_window(x0,y0,x,y)
        img = np.zeros(frame.shape, np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rect)

ret ,frame = cap.read()
if ret == True:
    cv2.imshow('image',frame)

while(wait):
    ret ,frame = cap.read()
    if ret == True:
        # frame = cv2.GaussianBlur(frame,(5,5),0)
        cv2.imshow('image',cv2.add(frame,img))
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
  help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
  help="max buffer size")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
# greenLower = (1, 200, 150)
# greenUpper = (10, 255, 255)
pts = deque(maxlen=args["buffer"])
 # [3, 229.0, 197.0]
# if a video path was not supplied, grab the reference
# to the webcam

 

# keep looping
while True:
  # grab the current frame
  (grabbed, frame) = cap.read()
 
  # if we are viewing a video and we did not grab a frame,
  # then we have reached the end of the video
  if args.get("video") and not grabbed:
    break
 
  # resize the frame, blur it, and convert it to the HSV
  # color space
  # frame = imutils.resize(frame, width=600)
  # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
  # construct a mask for the color "green", then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  mask = cv2.inRange(hsv, greenLower, greenUpper)
  # LPF for denoise?
  mask = cv2.erode(mask, None, iterations=2)
  mask = cv2.dilate(mask, None, iterations=2)
  cv2.imshow("mask", mask)

    # find contours in the mask and initialize the current
  # (x, y) center of the ball
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)[-2]
  center = None
 
  # only proceed if at least one contour was found
  if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
    # only proceed if the radius meets a minimum size
    if radius > 10:
      # draw the circle and centroid on the frame,
      # then update the list of tracked points
      cv2.circle(frame, (int(x), int(y)), int(radius),
        (0, 255, 255), 2)
      cv2.circle(frame, center, 5, (0, 0, 255), -1)
 
  # update the points queue
  pts.appendleft(center)
  # loop over the set of tracked points
  for i in xrange(1, len(pts)):
    # if either of the tracked points are None, ignore
    # them
    if pts[i - 1] is None or pts[i] is None:
      continue
 
    # otherwise, compute the thickness of the line and
    # draw the connecting lines
    thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
    cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
  # show the frame to our screen
  cv2.imshow("image", frame)
  key = cv2.waitKey(1) & 0xFF
 
  # if the 'q' key is pressed, stop the loop
  if key == ord("q"):
    break
 
# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()