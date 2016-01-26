from __future__ import division

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# calculate the scale that should be aplied to make the image
# fit into the window
def display_image(window_name, image):
    screen_res = 720, 480
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    # reescale the resolution of the window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # show image
    cv2.imshow(window_name, image)

    # wait for any key to quit the program
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# calculate the scale that should be aplied to make the image
# fit into the window
def display_frame(window_name, image):
    screen_res = 720, 480
    scale_width = screen_res[0] / image.shape[1]
    scale_height = screen_res[1] / image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(image.shape[1] * scale)
    window_height = int(image.shape[0] * scale)

    # reescale the resolution of the window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # show image
    cv2.imshow(window_name, image)


# fourier transform
def fourier(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


# calculate the histogram of the image
def histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist)
    # plt.hist(img_norm.ravel(),256,[0,256]);
    plt.show()


# find edge in a binarized image
def edge(image):
    edge_horizont = ndimage.sobel(image, 0)
    edge_vertical = ndimage.sobel(image, 1)
    magnitude = np.hypot(edge_horizont, edge_vertical)

    return magnitude


# binarize the image
def binarize(image):
    # binarize the image 0, 128, 255
    img_cpy = np.ndarray(shape=image.shape)

    # bom para edge
    #  64   0
    # 128 128
    # 255 255

    # bom para binarizacao
    #  32   0
    # 255 255

    # apply filter
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] <= 64:
                img_cpy[i][j] = 0
            elif image[i][j] <= 128:
                img_cpy[i][j] = 128
            elif image[i][j] <= 255:
                img_cpy[i][j] = 255

    return img_cpy


def binarize_02(image):
    ret, thresh1 = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY)
    return thresh1


# convert an norm image to grayscale image
def imgToGray(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return img_gray


# negative of an image
def negImage(image):
    img_neg = (255 - image);
    return img_neg


# read an image in normal mode
def readImage(filename):
    img = cv2.imread(filename)
    return img


# read an image in grayScale, dont check if file exists
def readGrayImage(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img
