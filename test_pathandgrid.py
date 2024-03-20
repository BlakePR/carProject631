from pathplanner import find_ave_angle
from pic2grid import crop_down, make_grid, grid2midpoints

import cv2 as cv
import numpy as np
import os


def test_find_ave_angle():
    obstacles = cv.imread("obstacles_image0.jpg")
    cropped = crop_down(obstacles, 120)
    grid = make_grid(cropped, 10, 10, 0.33)
    midpoints = grid2midpoints(grid, cropped.shape[1] // 10, cropped.shape[0] // 10)
    angle = find_ave_angle(midpoints)
    print(angle)
    assert angle == 0.0
