import cv2
import sys
import os
import subprocess

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
full_body = cv2.CascadeClassifier("haarcascade_fullbody.xml")
upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")


video_capture = cv2.VideoCapture(0)

text = "Welcome to Dreeshtee"

#os.system('.\\espeak.exe -s 100 "Welcome to Dreeshtee"')
#os.system('.\\espeak.exe  -g 20 -s 100 %(text)s'  % locals())

#Declaring a temp variable
temp_face = 0
temp_fullbody=0
temp_halfbody=0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #Capture from Photo
    #frame=cv2.imread("pic.jpg")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#face detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	#person full body detection
    person = full_body.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE
                                        )

	#Draw a recntangle around the full body of person
    for (ex, ey, ew, eh) in person:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
	
	#half body detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the halfbody
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
	# Display the resulting frame
    cv2.imshow('Video', frame)

    #Count the Number of faces
    count_face=len(faces)
	count_fullbody=len(faces)
	count_halfbody=len(faces)
	
 #   if count!=temp:
  #      os.system('.\\espeak.exe -s 100 %(count)s' % locals())

   # temp=count

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
