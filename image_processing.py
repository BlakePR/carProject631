import cv2 as cv
import numpy as np
import os

folder_dir = "/opt/ECEN_631/carProject631/pics/"
badfloor = f"{folder_dir}compiled_images/"

for images in sorted(os.listdir(badfloor)):
    image_RGB = cv.imread(badfloor + images)
    # cv.imshow("read_image", image_RGB)
    image_HSV = cv.cvtColor(image_RGB, cv.COLOR_BGR2HSV)
    #
    #Channel 0 and 1 could be useful if the current method doesn't work.
    # RGB_channel_0 = cv.extractChannel(image_RGB, 0)
    # RGB_thresh_2 = cv.threshold(RGB_channel_0, 20, 255, cv.THRESH_BINARY_INV)[1]
    # cv.imshow("RGB_thresl_0", RGB_thresh_2)
    # cv.imshow("RGB_channel_0",RGB_channel_0)

    # RGB_channel_1 = cv.extractChannel(image_RGB, 1)
    # RGB_thresh_2 = cv.threshold(RGB_channel_1, 50, 255, cv.THRESH_BINARY_INV)[1]
    # cv.imshow("RGB_thresl_1", RGB_thresh_2)
    # cv.imshow("RGB_channel_1", RGB_channel_1)

    RGB_channel_2 = cv.extractChannel(image_RGB, 2)
    RGB_thresh_2 = cv.threshold(RGB_channel_2,35,255,cv.THRESH_BINARY_INV)[1]
    RGB_thresh_22 = cv.threshold(RGB_channel_2, 125, 255, cv.THRESH_BINARY)[1]
    mask_RGB1 = cv.erode(RGB_thresh_2,None,iterations=1)
    mask_RGB1 = cv.dilate(mask_RGB1, None, iterations=4)
    cv.imshow("RGB_channel_2", RGB_channel_2)
    cv.imshow("RGB_thresh_2",mask_RGB1)
    cv.imshow("RGB_thresh_22", RGB_thresh_22)


    # This image doesn't seem readily usable for usable image information
    # HSV_channel_0 = cv.extractChannel(image_HSV, 0)
    # cv.imshow("HSV_channel_0", HSV_channel_0)

    HSV_channel_1 = cv.extractChannel(image_HSV, 1)
    #This threshold can be tuned in the future, but seems to work pretty well for the sample images
    threshold_HSV1 = cv.threshold(HSV_channel_1,120,255,cv.THRESH_BINARY)[1]
    mask_HSV1 = cv.erode(threshold_HSV1,None,iterations=1)
    mask_HSV1 = cv.dilate(mask_HSV1, None, iterations=5)
    cv.imshow("HSV_channel_1", HSV_channel_1)
    # cv.imshow("HSV_channel_1_thres", threshold_HSV1)
    cv.imshow("HSV_channel_1_mask", mask_HSV1)

    #Seems less reliable than other options
    # HSV_channel_2 = cv.extractChannel(image_HSV, 2)
    # threshold_HSV1 = cv.threshold(HSV_channel_2, 30, 255, cv.THRESH_BINARY_INV)[1]
    # cv.imshow("HSV_channel_2_thres", threshold_HSV1)
    # cv.imshow("HSV_channel_2", HSV_channel_2)

    mask_RGB1 = cv.bitwise_or(mask_RGB1,RGB_thresh_22)
    obstacles = cv.bitwise_and(mask_HSV1, mask_RGB1)
    cv.imshow("obstacles",obstacles)


    cv.waitKey(0)
