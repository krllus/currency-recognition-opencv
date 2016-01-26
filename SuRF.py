import cv2
import numpy as np

img = cv2.imread('files/img.png')

# Convert them to grayscale
imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SURF extraction
surf = cv2.xfeatures2d.SURF_create()
# kp, descritors = surf.detect(imgg,None,useProvidedKeypoints = False)
kp, descritors = surf.detectAndCompute(imgg, None)

# Setting up samples and responses for kNN
samples = np.array(descritors)
responses = np.arange(len(kp), dtype=np.float32)

# kNN training
knn = cv2.ml.KNearest_create()
# knn.train(samples,responses)
knn.train(samples, cv2.ml.ROW_SAMPLE, responses)

# Now loading a template image and searching for similar keypoints
template = cv2.imread('files/template/5_template.png')
templateg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
key, desc = surf.detectAndCompute(templateg, None)
# keys,desc = surf.detect(templateg,None,useProvidedKeypoints = False)

count = 0

for h, des in enumerate(desc):
    # des = np.array(des,np.float32).reshape((1,128))
    des = np.array(des, np.float32).reshape(1, len(des))
    # retval, results, neigh_resp, dists = knn.find_nearest(des,1)
    retval, results, neigh_resp, dists = knn.findNearest(des, 1)
    res, dist = int(results[0][0]), dists[0][0]

    if dist < 0.1:  # draw matched keypoints in red color
        color = (0, 0, 255)
        count += 1
    else:  # draw unmatched in blue color
        print dist
        color = (255, 0, 0)

    # Draw matched key points on original image
    x, y = kp[res].pt
    center = (int(x), int(y))
    cv2.circle(img, center, 2, color, -1)
    # if count==1:
    #    cv2.putText(img,">>BIG BOX<<", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(img, ">>BIG BOX<<", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

cv2.imshow('img', img)
cv2.imshow('tm', template)
cv2.waitKey(0)
cv2.destroyAllWindows()
