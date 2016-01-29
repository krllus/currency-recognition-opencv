#!/usr/bin/env python
import cv2
import numpy as np

from lib import glib

# global frame, captured by the webcam
frame = 0


def recognize_bill(descriptor, bill_name):
    global frame
    count = 0
    bill_center = (0, 0)
    for h, des in enumerate(descriptor):
        # des = np.array(des,np.float32).reshape((1,128))
        des = np.array(des, np.float32).reshape(1, len(des))
        # retval, results, neigh_resp, dists = knn.find_nearest(des,1)
        retval, results, neigh_resp, dists = knn.findNearest(des, 1)
        res, distance = int(results[0][0]), dists[0][0]

        x, y = kp[res].pt
        center = (int(x), int(y))

        if distance < 0.1:
            # draw matched keypoints in red color
            bill_center = center
            color = (0, 0, 255)
            count += 1
        else:
            # draw unmatched in blue color
            # print distance
            color = (255, 0, 0)

        # Draw matched key points on original image
        cv2.circle(frame, center, 2, color, -1)

    print float(count) / len(descriptor)

    # if 50% of the poins matches, write in the bill
    if float(count) / len(descriptor) >= 0.5:
        # cv2.putText(img,">>BIG BOX<<", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(frame, ">>" + bill_name + "<<", bill_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))


# SURF extraction
surf = cv2.xfeatures2d.SURF_create()
surf.setHessianThreshold(500)

print "opening templates images..."
# get the templates
# FRONT
temp_002_front = glib.readGrayImage("files/template/front/02.jpg")
temp_005_front = glib.readGrayImage("files/template/front/05.jpg")
temp_010_front = glib.readGrayImage("files/template/front/10.jpg")
temp_020_front = glib.readGrayImage("files/template/front/20.jpg")
temp_050_front = glib.readGrayImage("files/template/front/50.jpg")
temp_100_front = glib.readGrayImage("files/template/front/100.jpg")
# BACK
temp_002_back = glib.readGrayImage("files/template/back/02.jpg")
temp_005_back = glib.readGrayImage("files/template/back/05.jpg")
temp_010_back = glib.readGrayImage("files/template/back/10.jpg")
temp_020_back = glib.readGrayImage("files/template/back/20.jpg")
temp_050_back = glib.readGrayImage("files/template/back/50.jpg")
temp_100_back = glib.readGrayImage("files/template/back/100.jpg")

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
    frame
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

    # regognize bill acording to the descriptor
    recognize_bill(desc_002_back, "2 Reais Back")
    recognize_bill(desc_005_back, "5 Reais Back")
    recognize_bill(desc_010_back, "10 Reais Back")
    recognize_bill(desc_020_back, "20 Reais Back")
    recognize_bill(desc_050_back, "50 Reais Back")
    recognize_bill(desc_100_back, "100 Reais Back")

    recognize_bill(desc_002_front, "2 Reais Front")
    recognize_bill(desc_005_front, "5 Reais Front")
    recognize_bill(desc_010_front, "10 Reais Front")
    recognize_bill(desc_020_front, "20 Reais Front")
    recognize_bill(desc_050_front, "50 Reais Front")
    recognize_bill(desc_100_front, "100 Reais Front")

    # print a new line
    print "\n"

    # Display the result of the processing
    glib.display_frame("frame", frame)

    # run until press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # saving the frame image
        cv2.imwrite('frame.png', frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
