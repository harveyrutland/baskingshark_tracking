import numpy as np
import cv2 
from matplotlib import pyplot as plt
pair = cv2.imread('photo.png', 0)

def crop_left_half(image):
        cropped_img = image[0:1000, 0:640]
        return cropped_img
def crop_right_half(image):
    cropped_img = image[0:1000, 640:1280]
    return cropped_img


#Left

imgL = crop_left_half(pair)
#Right
imgR = crop_right_half(pair)

stereo = cv2.StereoBM_create(numDisparities=112, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()