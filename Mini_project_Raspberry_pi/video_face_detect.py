import cv2
from espeak import espeak

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
full_body = cv2.CascadeClassifier("haarcascade_fullbody.xml")
upper_body = cv2.CascadeClassifier("haarcascade_upperbody.xml")

video_capture = cv2.VideoCapture(0)

espeak.synth( "Welcome to Dreeshtee")

#Declaring a temp variable
temp = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #Capture from Photo
    #frame=cv2.imread("pic.jpg")
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


    person = full_body.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE
                                        )

    for (ex, ey, ew, eh) in person:
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the resulting frame
    #cv2.imshow('Video', frame)

    #Count the Number of faces
    count=len(faces)

    #espeak code to tell the number of persons in front 
    if count!=temp:
        espeak.synth('%(count)s person within 1 meter' % locals())

    temp=count

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
