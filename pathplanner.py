from pic2grid import crop_down, make_grid, grid2midpoints

import cv2 as cv
import numpy as np


def find_ave_angle(midpoints):
    angles = []
    dist = 6
    weighting = 0
    for i in range(dist):
        weighting +=1
        curr_pt = midpoints[-1 - i]
        next_pt = midpoints[-2 - i]
        dely = next_pt[0] - curr_pt[0]
        delx = next_pt[1] - curr_pt[1]
        angle = np.arctan2(delx, -dely)
        angles.append(angle)

    sum = np.sum(angles) / weighting
    return np.rad2deg(sum) / 3.0