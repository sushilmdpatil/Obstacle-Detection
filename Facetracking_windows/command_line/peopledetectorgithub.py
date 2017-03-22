# Pedestrian Detection Default
# use opencv
import cv2


# The HOG detector returns slightly larger rectangles than the real objects.
# So we slightly shrink the rectangles to get a nicer output.
def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


# Default detector (using INRIA Person Dataset)
    image = cv2.imread("pic.jpg")  # read image
    hog = cv2.HOGDescriptor()  # derive HOG features
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # SVM

    # pedestrian detection
    found, w = hog.detectMultiScale(image, hitThreshold=0, winStride=(8, 8), padding=(0, 0), scale=1.05,
                                    finalThreshold=2)
    draw_detections(image, found)  # draw rectangles

    # write & save image
    cv2.imshow('original', image)  # write image
    cv2.waitKey(0)  # for keyboard binding
    cv2.imwrite('test2.jpg',im) # save image
    cv2.destroyAllWindows() # clean up
