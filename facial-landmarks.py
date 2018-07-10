# import the necessary packages
from scipy.spatial import distance as dist
import scipy.misc
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from scipy.signal import argrelextrema
import time
from collections import deque
import math
from scipy import interpolate

def eye_aspect_ratio(eye):
  # compute the euclidean distances between the two sets of
  # vertical eye landmarks (x, y)-coordinates
  A = dist.euclidean(eye[1], eye[5])
  B = dist.euclidean(eye[2], eye[4])
 
  # compute the euclidean distance between the horizontal
  # eye landmark (x, y)-coordinates
  C = dist.euclidean(eye[0], eye[3])
 
  # compute the eye aspect ratio
  ear = (A + B) / (2.0 * C)
 
  # return the eye aspect ratio
  return ear

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):

  overlay = image.copy()
  output = image.copy()
  if colors is None:
    colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
      (168, 100, 168), (158, 163, 32),
      (163, 38, 32), (180, 42, 220)]
  # loop over the facial landmark regions individually
  for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
    # grab the (x, y)-coordinates associated with the
    # face landmark
    (j, k) = FACIAL_LANDMARKS_IDXS[name]
    pts = shape[j:k]
 
    # check if are supposed to draw the jawline
    if name == "jaw":
      # since the jawline is a non-enclosed facial region,
      # just draw lines between the (x, y)-coordinates
      for l in range(1, len(pts)):
        ptA = tuple(pts[l - 1])
        ptB = tuple(pts[l])
        cv2.line(overlay, ptA, ptB, colors[i], 2)
 
    # otherwise, compute the convex hull of the facial
    # landmark coordinates points and display it
    else:
      hull = cv2.convexHull(pts)
      cv2.drawContours(overlay, [hull], -1, colors[i], -1)
  # apply the transparent overlay
  cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
 
  # return the output image
  return output

def process_eye(eye,pos):
  global gray
  area = cv2.contourArea(eye)
  x,y,w,h = cv2.boundingRect(eye)
  x = int(x)
  y = int(y-0.25*h) 
  w = int(w)
  h = int(1.5*h)
  eye_image = gray[y:y+h,x:x+w]
  cv2.imshow(pos,eye_image)
  i = 0
  r = np.float64(eye_image.max()-eye_image.min())
  r = 255/r
  u = np.zeros(eye_image.shape,np.float64)
  u = r*(eye_image - eye_image.min())
  pixelpoints = np.transpose(u)
  delta = np.float64(u[:,:1].mean() - u[:,u.shape[1]-1:].mean())/u.shape[1]
  for row in pixelpoints:
    # correct acording to nose shadow
    # TODO: adjust correction function
    m = row.mean() + delta*(i-u.shape[1])
    u[:,i] = m if m>0 else 0
    i = i + 1

  # expand range again ??
  u = 255*(u -u.min())/(u.max()-u.min())

  dr = u[0,:].copy()
  for bx in range(u.shape[1]-1):
    dr[bx] = (bx-u.shape[1]/2)*(255-dr[bx])
  coefs2 = np.polyfit(list(range(u.shape[1])),dr,5)
  r3 = np.zeros(200,np.float64)
  ffit = np.poly1d(coefs2)
  for bx in range(200):
    i = np.float64(bx)*u.shape[1]/200
    m = ffit(i)
    r3[bx] = m

  right = r3.mean()
  # u = np.uint8(r)
  return (x,y,w,h),right



# blink detection initialization
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
#needs to download predictor model
#TODO: try shape_predictor_5_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# FACIAL_LANDMARKS_IDXS = OrderedDict([
#   ("mouth", (48, 68)),
#   ("right_eyebrow", (17, 22)),
#   ("left_eyebrow", (22, 27)),
#   ("right_eye", (36, 42)),
#   ("left_eye", (42, 48)),
#   ("nose", (27, 35)),
#   ("jaw", (0, 17))
# ])
# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream("eye.avi").start()
# fileStream = True
cap = cv2.VideoCapture(0)
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
tracker = cv2.TrackerKCF_create()
# tracker = cv2.TrackerBoosting_create()
ret,frame = cap.read()
# frame = imutils.resize(frame, width=450)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
t0 = 0
try:
  c,r,w,h = faces[0]
  face = (c,r,w,h)
  # print face
  # exit()
  ok = tracker.init(frame, face)
except:
  pass

pos = deque(maxlen=8)
max_r = 0
min_r = 0
val_range_l = (0,0) # max/min for left eye
val_range_r = (0,0) # max/min for right eye

# loop over frames from the video stream
while True:

  ret,frame = cap.read()
  # frame = cv2.GaussianBlur(frame,(7,7),0)

  # frame = imutils.resize(frame, width=450)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  # gray = clahe.apply(gray)

  rects = dlib.rectangles()
  # (self: dlib.rectangle, left: int, top: int, right: int, bottom: int
  dlib.rectangles.append(rects,dlib.rectangle(0,0,0,0))



  # detect faces in the grayscale frame
  rects = detector(gray, 0)
  if not len(rects)>0:
    try:
      print "Use tracker data"
      ok, face = tracker.update(frame)

      if ok:
        # Tracking success
        print "success"
        face = (int(face[0]),int(face[1]),int(face[2]),int(face[3]))
        c,r,w,h = face
        rects = dlib.rectangles()
        dlib.rectangles.append(rects,dlib.rectangle(c,r,c+w,r+h))
      else :
        print "error"
        # Tracking failure
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        c,r,w,h = faces[0]
        face = (c,r,w,h)  
        rects = dlib.rectangles()
        dlib.rectangles.append(rects,dlib.rectangle(c,r,c+w,r+h))
    except:
      print "except"
  # else:
  #   ok, face = tracker.update(frame)


  # print(face_rect)

# rectangles[[(280, 194) (538, 452)]]
# [(261, 162), (564, 465)]

  # loop over the face detections
  for rect in rects:
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # loop over the face parts individually
    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
      # clone the original image so we can draw on it, then
      # display the name of the face part on the image
      clone = frame.copy()
      cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 255), 2)
   
      # loop over the subset of facial landmarks, drawing the
      # specific face part
      for (x, y) in shape[i:j]:
        cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

    output = face_utils.visualize_facial_landmarks(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR), shape)
    cv2.imshow("output", output)

    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
 
    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # compute the convex hull for the left and right eye, then
    # visualize each of the eyes
    (x,y,w,h),right_l = process_eye(leftEye,"left")

    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow("band",u)
    cv2.putText(frame,"FPS:" + str(1/(time.time()-t0)), (10, 150),
      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    t0 = time.time()
    (x,y,w,h),right_r= process_eye(rightEye,"right")


    pos.append((right_l,right_r))
    avg = (pos[0][0],pos[0][1])
    for i in xrange(1, len(pos)):
        avg = ((avg[0]+pos[i][0]), (avg[1]+pos[i][1]))
    avg = (avg[0]/len(pos), avg[1]/len(pos))

    val_range_l = (max(val_range_l[0],right_l),min(val_range_l[1],right_l))
    rel_l =  (avg[0]-val_range_l[1])/(val_range_l[0]-val_range_l[1])*100

    val_range_r = (max(val_range_r[0],right_r),min(val_range_r[1],right_r))
    rel_r =  (avg[0]-val_range_r[1])/(val_range_r[0]-val_range_r[1])*100

    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(frame, str(rel_l), (10, 90),
      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, str(rel_r), (10, 120),
      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.circle(frame,(320,63), 15, 50, -1)
    center2 = (640-(rel_l+rel_r)/2*640/100)
    center2 = int(center2)
    cv2.circle(frame,(center2,63), 20, 0, -1)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    if ear < EYE_AR_THRESH:
      COUNTER += 1
    else:
      if COUNTER >= EYE_AR_CONSEC_FRAMES:
        TOTAL += 1
      COUNTER = 0

    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
  # show the frame
  # cv2.namedWindow("Frame", cv2.WND_PROP_FULLSCREEN)
  # cv2.setWindowProperty("Frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN) 
  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1) & 0xFF
 
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break
 
# do a bit of cleanup
# cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()