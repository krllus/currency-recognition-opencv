#!/usr/bin/env python
import cv2
import numpy as np

from lib import glib


def my_awesome_function():
    # TODO finish this awesome function where all the calculations will be performed
    print "this function is awesome"


# SURF extraction
surf = cv2.xfeatures2d.SURF_create()

print "opening templates images..."
# get the templates
# FRONT
temp_002_front = glib.readGrayImage("files/template/front/05.jpg")
temp_005_front = glib.readGrayImage("files/template/front/05.jpg")
temp_010_front = glib.readGrayImage("files/template/front/05.jpg")
temp_020_front = glib.readGrayImage("files/template/front/20.jpg")
temp_050_front = glib.readGrayImage("files/template/front/05.jpg")
temp_100_front = glib.readGrayImage("files/template/front/05.jpg")
# BACK
temp_002_back = glib.readGrayImage("files/template/back/05.jpg")
temp_005_back = glib.readGrayImage("files/template/back/05.jpg")
temp_010_back = glib.readGrayImage("files/template/back/05.jpg")
temp_020_back = glib.readGrayImage("files/template/back/20.jpg")
temp_050_back = glib.readGrayImage("files/template/back/05.jpg")
temp_100_back = glib.readGrayImage("files/template/back/05.jpg")

# get keypoints and descriptors for each template
# FRONT
print "calculating template descriptors..."
key_01, desc_002_front = surf.detectAndCompute(temp_002_front, None)
key_02, desc_005_front = surf.detectAndCompute(temp_005_front, None)
key_03, desc_010_front = surf.detectAndCompute(temp_010_front, None)
key_04, desc_020_front = surf.detectAndCompute(temp_020_front, None)
key_05, desc_050_front = surf.detectAndCompute(temp_050_front, None)
key_06, desc_100_front = surf.detectAndCompute(temp_100_front, None)
# BACK
key_07, desc_002_back = surf.detectAndCompute(temp_002_back, None)
key_08, desc_005_back = surf.detectAndCompute(temp_005_back, None)
key_09, desc_010_back = surf.detectAndCompute(temp_010_back, None)
key_10, desc_020_back = surf.detectAndCompute(temp_020_back, None)
key_11, desc_050_back = surf.detectAndCompute(temp_050_back, None)
key_12, desc_100_back = surf.detectAndCompute(temp_100_back, None)

print "starting video capture..."
# start the video capture
cap = cv2.VideoCapture(0)

# kNN creation
knn = cv2.ml.KNearest_create()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # transform the frame to gray
    gray = glib.imgToGray(frame)

    # surf descriptors for the frame
    kp, desc = surf.detectAndCompute(gray, None)

    # Setting up samples and responses for kNN
    samples = np.array(desc)
    responses = np.arange(len(kp), dtype=np.float32)

    # kNN training
    knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # TODO for each template, test if has found a correspondent bill
    my_awesome_function()

    # Display the result of the processing
    glib.display_frame("frame", frame)

    # run until press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
