import cv2
import sys
import os
import subprocess

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


text = 'Hi Welcome to Drishti'

#os.system('.\\espeak.exe  -g 20 -s 100 %(text)s'  %locals())

#Capture from Photo
frame=cv2.imread("pic1.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


    # Draw a rectangle around the faces
for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


eyes = eye_cascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE
                                        )

for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

found, w = hog.detectMultiScale(gray,
                                winStride=(8, 8),
                                padding=(32, 32),
                                scale=1.05)

for (bx,by,bw,bh) in found:
    cv2.rectangle(gray, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)

    # Display the resulting frame
cv2.imshow('Video', frame)

    #Count the Number of faces
count=len(found)

os.system('.\\espeak.exe -s 100 %(count)s' % locals())


cv2.waitKey(0)

# When everything is done, release the capture
cv2.destroyAllWindows()