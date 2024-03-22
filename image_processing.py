import cv2 as cv
import numpy as np
import os

folder_dir = "pics/"
badfloor = "imseries/"

def get_wall_HS_mask(rgbimg):
    image_hsv = cv.cvtColor(rgbimg, cv.COLOR_BGR2HSV)
    
    hue = cv.extractChannel(image_hsv, 0)
    saturation = cv.extractChannel(image_hsv, 1)
    intensity = cv.extractChannel(image_hsv, 2)

    RGB_channel_2 = cv.extractChannel(rgbimg, 2)
    RGB_channel_1 = cv.extractChannel(rgbimg, 1)
    RGB_channel_0 = cv.extractChannel(rgbimg, 0)

    #Right wall (bright wall)
    RGB_thresh_2 = cv.threshold(RGB_channel_2, 150, 255, cv.THRESH_BINARY)[1]
    RGB_thresh_2_inv = cv.threshold(RGB_channel_2, 190, 255, cv.THRESH_BINARY_INV)[1]
    RGB_thresh_tot_2 = cv.bitwise_and(RGB_thresh_2, RGB_thresh_2_inv)
    RGB_thresh_1 = cv.threshold(RGB_channel_1, 120, 255, cv.THRESH_BINARY)[1]
    RGB_thresh_1_inv = cv.threshold(RGB_channel_1, 180, 255, cv.THRESH_BINARY_INV)[1]
    RGB_thresh_tot_1 = cv.bitwise_and(RGB_thresh_1, RGB_thresh_1_inv)
    color_coordinate_bright = cv.bitwise_and(RGB_thresh_tot_2, RGB_thresh_tot_1)

    #Dark wall
    RGB_thresh_2 = cv.threshold(RGB_channel_2, 110, 255, cv.THRESH_BINARY)[1]
    RGB_thresh_2_inv = cv.threshold(RGB_channel_2, 140, 255, cv.THRESH_BINARY_INV)[1]
    RGB_thresh_tot_2 = cv.bitwise_and(RGB_thresh_2, RGB_thresh_2_inv)
    RGB_thresh_1 = cv.threshold(RGB_channel_1, 70, 255, cv.THRESH_BINARY)[1]
    RGB_thresh_1_inv = cv.threshold(RGB_channel_1, 100, 255, cv.THRESH_BINARY_INV)[1]
    RGB_thresh_tot_1 = cv.bitwise_and(RGB_thresh_1, RGB_thresh_1_inv)
    color_coordinate_dark = cv.bitwise_and(RGB_thresh_tot_2, RGB_thresh_tot_1)

    color_coordinate = cv.bitwise_or(color_coordinate_bright, color_coordinate_dark)


    hue_mask_for_walls_high = cv.threshold(hue, 100, 255, cv.THRESH_BINARY)[1]
    hue_mask_for_walls_low = cv.threshold(hue, 25, 255, cv.THRESH_BINARY_INV)[1]

    sat_mask_high = cv.threshold(saturation, 100, 255, cv.THRESH_BINARY)[1]
    sat_mask_low = cv.threshold(saturation, 25, 255, cv.THRESH_BINARY_INV)[1]

    wall1 = cv.bitwise_and(hue_mask_for_walls_high,sat_mask_low)
    wall2 = cv.bitwise_and(hue_mask_for_walls_low,sat_mask_high)
    sat_mask = cv.bitwise_or(wall1, wall2)
    
    hsv_mask = sat_mask
    # hsv_mask = cv.bitwise_and(int_mask, sat_mask)

    hue_mask = cv.threshold(hue, 25, 255, cv.THRESH_BINARY_INV)[1]
    hue_mask2 = cv.threshold(hue, 160, 255, cv.THRESH_BINARY)[1]
    hue_mask = cv.bitwise_or(hue_mask, hue_mask2)

    int_mask = cv.threshold(intensity, 200, 255, cv.THRESH_BINARY_INV)[1]
    int_mask_2 = cv.threshold(intensity, 100, 255, cv.THRESH_BINARY)[1]
    int_mask = cv.bitwise_and(int_mask, int_mask_2)


    and_pic = cv.bitwise_or(color_coordinate, hsv_mask)
    
    final_pic = and_pic
    final_pic = cv.bitwise_and(and_pic, hue_mask)
    final_pic = cv.bitwise_and(final_pic, int_mask)
    final_pic = cv.bitwise_and(final_pic, cv.threshold(saturation, 25, 255, cv.THRESH_BINARY)[1])
    final_pic = cv.bitwise_and(final_pic, cv.threshold(RGB_channel_0, 200, 255, cv.THRESH_BINARY_INV)[1])

    final_pic = cv.erode(final_pic, None, iterations=5)
    final_pic = cv.dilate(final_pic, None, iterations=10)
    # cv.imshow("color_coordinate",color_coordinate)
    # cv.imshow("hsv_mask",hsv_mask)
    # cv.imshow("and_pic",and_pic)
    # cv.imshow("final_pic",final_pic)

    # cv.imshow("image_RGB",rgbimg)
    # cv.imshow("image_hsv",image_hsv)

    return final_pic


def get_noodle_not_red(bgrimg):
    chanel2 = cv.extractChannel(bgrimg, 2)
    thresh = cv.threshold(chanel2, 40, 255, cv.THRESH_BINARY_INV)[1]
    mask = cv.erode(thresh, None, iterations=1)
    mask = cv.dilate(mask, None, iterations=4)
    return mask


def get_obstacle(rgbimg):
    wall_mask = get_wall_HS_mask(rgbimg)
    middle = np.zeros_like(wall_mask)
    middle[:, 3*middle.shape[1]//8:5*middle.shape[1]//8] = 1
    middle = cv.bitwise_not(middle)
    wall_mask = cv.bitwise_and(wall_mask, middle)
    noodle_mask = get_noodle_not_red(rgbimg)
    obstacles = cv.bitwise_or(wall_mask, noodle_mask)

    return obstacles

if __name__ == "__main__":
    for images in sorted(os.listdir(badfloor)):
        image_RGB = cv.imread(badfloor + images)
        obstacles = get_obstacle(image_RGB)
        cv.imshow("image_RGB",image_RGB)
        cv.imshow("obstacles",obstacles)
        cv.waitKey(0)

