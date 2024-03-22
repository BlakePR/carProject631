import cv2 as cv
import numpy as np
import os

folder_dir = "pics/"
badfloor = "imseries/"


def get_noodle_hue_mask(hsvimg):
    hue = cv.extractChannel(hsvimg, 0)
    hue_mask = cv.inRange(hue, 85, 96)
    hue_mask = cv.erode(hue_mask, None, iterations=3)
    hue_mask = cv.dilate(hue_mask, None, iterations=6)
    return hue_mask


def get_wall_HS_mask(rgbimg):
    image_hsv = cv.cvtColor(rgbimg, cv.COLOR_BGR2HSV)
    hue = cv.extractChannel(image_hsv, 0)
    saturation = cv.extractChannel(image_hsv, 1)
    intensity = cv.extractChannel(image_hsv, 2)

    RGB_channel_2 = cv.extractChannel(rgbimg, 2)
    RGB_channel_1 = cv.extractChannel(rgbimg, 1)
    RGB_channel_0 = cv.extractChannel(rgbimg, 0)


    RGB_thresh_1 = cv.threshold(RGB_channel_1, 120, 255, cv.THRESH_BINARY)[1]
    RGB_thresh_1_inv = cv.threshold(RGB_channel_1, 150, 255, cv.THRESH_BINARY_INV)[1]
    RGB_thresh_tot_1 = cv.bitwise_and(RGB_thresh_1, RGB_thresh_1_inv)

    int_mask = cv.threshold(intensity, 155, 255, cv.THRESH_BINARY_INV)[1]
    int_mask_2 = cv.threshold(intensity, 130, 255, cv.THRESH_BINARY)[1]
    int_mask = cv.bitwise_and(int_mask, int_mask_2)
    sat_mask = cv.threshold(saturation, 20, 255, cv.THRESH_BINARY)[1]
    hsv_mask = cv.bitwise_and(int_mask, sat_mask)

    hue_mask = cv.threshold(hue, 35, 255, cv.THRESH_BINARY_INV)[1]
    hue_mask2 = cv.threshold(hue, 10, 255, cv.THRESH_BINARY)[1]
    hue_mask = cv.bitwise_and(hue_mask, hue_mask2)


    and_pic = cv.bitwise_or(RGB_thresh_tot_1, hsv_mask)
    final_pic = cv.bitwise_and(and_pic, hue_mask)
    final_pic = cv.erode(final_pic, None, iterations=2)
    final_pic = cv.dilate(final_pic, None, iterations=6)
    return final_pic


def get_noodle_not_red(bgrimg):
    chanel2 = cv.extractChannel(bgrimg, 2)
    thresh = cv.threshold(chanel2, 20, 255, cv.THRESH_BINARY_INV)[1]
    mask = cv.erode(thresh, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=4)
    return mask


def get_obstacle(rgbimg):
    wall_mask = get_wall_HS_mask(rgbimg)
    noodle_mask = get_noodle_not_red(rgbimg)
    obstacles = cv.bitwise_or(wall_mask, noodle_mask)

    return obstacles

if __name__ == "__main__":
    for images in sorted(os.listdir(badfloor)):
         image_RGB = cv.imread(badfloor + images)
         obstacles = get_obstacle(image_RGB)
         cv.imshow("obstacles", obstacles)
         cv.waitKey(0)
