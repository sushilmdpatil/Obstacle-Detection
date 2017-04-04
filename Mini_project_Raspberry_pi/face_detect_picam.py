import cv2
from espeak import espeak
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
full_body = cv2.CascadeClassifier("haarcascade_fullbody.xml")
upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)
temp=0

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
    

        #Count the Number of faces
	count=len(faces)
    
        #espeak code to tell the number of persons in front 
	if count!=temp:
		espeak.synth('%(count)s person within 1 meter' % locals())

	temp=count

        # show the frame
	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) & 0xFF
	
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
# When everything is done, release the capture
cv2.destroyAllWindows()
