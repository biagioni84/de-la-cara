import numpy as np
import cv2

def auto_canny(image, sigma=0.33):
  # compute the median of the single channel pixel intensities
  v = np.median(image)
 
  # apply automatic Canny edge detection using the computed median
  lower = int(max(0, (1.0 - sigma) * v))
  upper = int(min(255, (1.0 + sigma) * v))
  edged = cv2.Canny(image, lower, upper)
 
  # return the edged image
  return edged

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (150,150))

cap = cv2.VideoCapture(0)
# take first frame of the video
ret,img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
# tracker = cv2.TrackerKCF_create()
tracker = cv2.TrackerBoosting_create()
try:
  c,r,w,h = faces[0]
  face = (c,r,w,h)
  # print face
  # exit()
  ok = tracker.init(img, face)
except:
  pass

while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)


    ok, face = tracker.update(img)
    if ok:
      # Tracking success
      p1 = (int(face[0]), int(face[1]))
      p2 = (int(face[0] + face[2]), int(face[1] + face[3]))
      cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(img, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        c,r,w,h = faces[0]
        face = (c,r,w,h)  

    # faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # c,r,w,h = faces[0]
    # face = (c,r,w,h)

    #TODO: use trackers for the eyes
    x,y,w,h = int(face[0]),int(face[1]),int(face[2]),int(face[3])
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
         eye = roi_gray[ey:ey+eh, ex:ex+ew]
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    right_eye = right_eye_cascade.detectMultiScale(roi_gray,1.05,5)
    for (ex,ey,ew,eh) in right_eye:
         r_eye = roi_gray[ey:ey+eh, ex:ex+ew]
         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    # left_eye = left_eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in left_eye:
    #      cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)       
    # print r_eye.shape
    # exit()
    cv2.imshow('img',img)
    try:
      r_eye = cv2.equalizeHist(r_eye)

      # th_eye = cv2.fastNlMeansDenoising(r_eye)
      # th_eye = cv2.adaptiveThreshold(r_eye,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
      #       cv2.THRESH_BINARY,11,2) 
      # cv2.imshow('theye',th_eye)

    except NameError:
      print "no right eye"

    # windowClose = np.ones((5,5),np.uint8)
    # windowOpen = np.ones((2,2),np.uint8)
    # windowErode = np.ones((2,2),np.uint8)
    # ret, pupilFrame = cv2.threshold(r_eye,55,255,cv2.THRESH_BINARY)    #50 ..nothin 70 is better
    # pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
    # pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
    # pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)
    # ret,r_eye = cv2.threshold(r_eye,60,255,cv2.THRESH_BINARY)

    

    # try :
    #   circles = cv2.HoughCircles(r_eye,cv2.HOUGH_GRADIENT,1,80,
    #                             param1=50,param2=30,minRadius=0,maxRadius=0)
    #   circles = np.uint16(np.around(circles))
    #   for i in circles[0,:]:
    #       # draw the outer circle
    #       cv2.circle(r_eye,(i[0],i[1]),i[2],(0,255,0),2)
    #       # draw the center of the circle
    #       cv2.circle(r_eye,(i[0],i[1]),2,(0,0,255),3)
    # except:
    #   pass

    try :
      out= cv2.copyMakeBorder(r_eye,0,150-r_eye.shape[0],0,150-r_eye.shape[1],cv2.BORDER_CONSTANT,255)
      out.write(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
      constant = out[0:70,0:70]
      # constant = cv2.GaussianBlur(constant,(3,3),0)
      # constant = cv2.bilateralFilter(constant,9,75,75)

      th2 = cv2.adaptiveThreshold(constant,0,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
      neg = 255-th2

      kernel = np.ones((3,3),np.uint8)
      erosion = cv2.erode(neg,kernel,iterations = 1)
      opening = cv2.morphologyEx(neg, cv2.MORPH_OPEN, kernel)
      closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
      th2 = 255-closing


      edges = auto_canny(th2)

      cv2.imshow('edges',edges)
      cv2.imshow('th2',th2)
      # cv2.imshow('img2_fg',img2_fg)
    except:
      pass

    k = cv2.waitKey(1) & 0xff
    if k == 27 : break


cap.release()
out.release()
# cv2.imshow('img',img)
# cv2.waitKey(0)
cv2.destroyAllWindows()