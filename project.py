#!/usr/bin/env python
from lib import glib

# sebir http://www.g7smy.co.uk/?p=366

# open the image in gray scale
img_gray = glib.readGrayImage("files/bills/20_02.jpg")

# convert img to binary mode, black or white
img_bnry = glib.binarize(img_gray)

glib.display_image('binary',img_bnry)

# get the templates
# get keypoints and descriptors for each template

# start the video capture
    # for each frame,
    # try to match the templates
    # if one of them matches,
        # show results, go next frame or next template?
        #