import cv2 as cv
import numpy as np
import os

# Load the image
img = cv.imread("obstacles_image0.jpg")

cv.imshow("Original Image", img)
cv.waitKey(0)

imsize = img.shape
print("Image Size: ", imsize)


def crop_down(image, crop_height):
    return image[crop_height:, :, :]


cropped = crop_down(img, 120)
cv.imshow("Cropped Image", cropped)
cv.waitKey(0)


def make_grid(image, n_rows, n_cols, p_occup):
    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    thresh = p_occup * 255.0
    dx = image.shape[1] // n_cols
    dy = image.shape[0] // n_rows
    for i in range(n_rows):
        for j in range(n_cols):
            y = i * dy
            x = j * dx
            roi = image[y : y + dy, x : x + dx]
            mean = cv.mean(roi)
            if mean[0] > thresh:
                grid[i, j] = 1
    return grid


grid = make_grid(cropped, 10, 10, 0.33)
grid_big = cv.resize(
    grid, None, fx=imsize[1] // 10, fy=imsize[0] // 10, interpolation=cv.INTER_NEAREST
)
grid_big *= 250
cv.imshow("Grid", grid_big)
cv.waitKey(0)


def grid2midpoints(grid):
    midpoints = []
    for i in range(grid.shape[0]):  # rows
        jl = grid.shape[1] // 2 - 1
        jr = grid.shape[1] // 2
        while jl >= 0 and grid[i, jl] == 0:
            jl -= 1
        while jr < grid.shape[1] and grid[i, jr] == 0:
            jr += 1
        templ = float(jl)
        tempr = float(jr)
        midpoints.append((i, (templ + tempr) / 2))
    return midpoints
