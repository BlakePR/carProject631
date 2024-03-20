from pathplanner import find_ave_angle
from pic2grid import crop_down, crop_up, make_grid, grid2midpoints
import image_processing

import cv2 as cv
import numpy as np
import os


def test_find_ave_angle(image):
    frame = cv.imread(image)

    cropped = crop_down(frame, 120)
    cropped = crop_up(cropped, 30)
    obstacles = image_processing.get_obstacle(cropped)
    cv.imshow("obstacles", obstacles)
    grid = make_grid(obstacles, 10, 10, 0.33)
    cv.imshow("grid", grid*250)
    midpoints = grid2midpoints(grid, cropped.shape[1] // 10, cropped.shape[0] // 10)
    angle = find_ave_angle(midpoints)
    print(angle)
    cv.imshow("the actual frame", frame)
    cv.waitKey(0)

if __name__ == '__main__':
    test_find_ave_angle('image0.jpg')
    test_find_ave_angle('image1.jpg')
    test_find_ave_angle('image2.jpg')
    test_find_ave_angle('image3.jpg')

