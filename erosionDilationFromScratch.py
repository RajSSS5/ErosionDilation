import cv2
import numpy as np
from dataPath import DATA_PATH
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

imageName = DATA_PATH + "images/dilation_example.jpg"


def dilate(im, element):
    ksize = element.shape[0]
    height,width = im.shape[:2]

    border = ksize//2
    # Threshold image
    
    if(len(im.shape) > 2):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY)

    # Create a padded image with zeros padding
    paddedIm = np.zeros((height + border*2, width + border*2))
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
    for h_i in range(border, height+border):
        for w_i in range(border,width+border):
            # When you find a white pixel
            if im[h_i-border,w_i-border] == 255:
                paddedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1] = 255
    return paddedIm[border:height+border,border:width+border] 

def erode(im, element):
    ksize = element.shape[0]
    height,width = im.shape[:2]
    
    border = ksize//2

    # Threshold image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
    ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    
    # Create a padded image with zeros padding
    paddedIm = np.full((height + border*2, width + border*2),255)
    paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 255)
    for h_i in range(border, height+border):
        for w_i in range(border,width+border):
            # When you find a black pixel
            if im[h_i-border,w_i-border] == 0:
                paddedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1] = \
                        cv2.bitwise_and(paddedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1],element)
    return paddedIm[border:height+border,border:width+border]


# Read the input image
image = cv2.imread(imageName)

# # Check for an invalid input
if image is None:
    print("Could not open or find the image")
    exit()


kSize = (3,3)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)

# Apply dilate function on the input image
imageDilated = dilate(image, kernel1)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Original Image")
plt.subplot(122);plt.imshow(imageDilated);plt.title("Dilated Image")

plt.show()

# Get structuring element/kernel which will be used for dilation
kSize = (3,3)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kSize)

# Image taken as input
imageName = DATA_PATH + "images/erosion_example.jpg"
image = cv2.imread(imageName, cv2.IMREAD_COLOR)

# Check for invalid input
if image is None:
    print("Could not open or find the image")
    exit()

# Eroding the image , decreases brightness of image
imageEroded = erode(image, kernel2)

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(image);plt.title("Original Image")
plt.subplot(122);plt.imshow(imageEroded);plt.title("Eroded Image");
plt.show()