import cv2
import sys
import os
import subprocess

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
full_body = cv2.CascadeClassifier("haarcascade_fullbody.xml")
upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")


video_capture = cv2.VideoCapture(0)
var='y'
while var=='y':
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if ret:
        #Capture from Photo
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

        h_body = upper_body.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        # Draw a rectangle around the halfbody
        for (x, y, w, h) in h_body:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        cv2.waitKey(10)
        var=input("Do you want to contnue y/n?")
        if(var=='y'):
                continue
        else:
                break
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
    else:
        print('Frame not captured properly')
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
