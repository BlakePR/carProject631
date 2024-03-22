# python3 ECEnRacer.py
""" 
This program is for ECEN-631 BYU Race
*************** RealSense Package ***************
From the Realsense camera:
	RGB Data
	Depth Data
	Gyroscope Data
	Accelerometer Data
*************** Arduino Package ****************
	Steer(int degree) : -30 (left) to +30 (right) degrees
	Drive(float speed) : -3.0 to 3.0 meters/second
	Zero(int PWM) : Sets front wheels going straight around 1500
	Encoder() : Returns current encoder count.  Reset to zero when stop
	Pid(int flag) : 0 to disable PID control, 1 to enable PID control
	KP(float p) : Proporation control 0 ~ 1.0 : how fast to reach the desired speed.
	KD(float d) : How smoothly to reach the desired speed.

    EXTREMELY IMPORTANT: Read the user manual carefully before operate the car
**************************************
"""

# import the necessary packages
from Arduino import Arduino
from RealSense import *
import numpy as np
import imutils
import cv2

from pic2grid import crop_down, crop_up, make_grid, grid2midpoints
from pathplanner import find_ave_angle
from image_processing import get_obstacle

rs = RealSense("/dev/video2", RS_VGA)  # RS_VGA, RS_720P, or RS_1080P
writer = None

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)  # Linux
# Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac
Car.pid(1)  # Use PID control
# You can use kd and kp commands to change KP and KD values.  Default values are good.
# loop over frames from Realsense
count = 0
while True:
    (time, rgb, depth, accel, gyro) = rs.getData()

    # cv2.imshow("RGB", rgb)
    # cv2.imshow("Depth", depth)

    Car.zero(1576)  # Set car to go straight.  Change this for your car.

    """
    Add your code to process rgb, depth, IMU data
    """
    crop = crop_down(rgb, 120)
    crop = crop_up(crop, 30)
    crop = get_obstacle(crop)
    gridx, gridy = 15, 16
    grid = make_grid(crop, gridx, gridy, 0.35)
    scalex = crop.shape[1] // gridx
    scaley = crop.shape[0] // gridy
    midpoints = grid2midpoints(grid, scalex=scalex, scaley=scaley)
    angle = find_ave_angle(midpoints)
    """
    Control the Car
    """
    count += 1
    Car.steer(angle)
    if count < 40:
        Car.drive(1.8)
    else:
        Car.drive(1.3)
    print(angle)
    if count % 13 == 0:
        cv2.imwrite(f"run2_{count}_crop.jpg", crop)
        cv2.imwrite(f"run2_{count}.jpg", rgb)
    if count > 160:
        break
    """
   	IMPORTANT: Never go full speed. Use CarTest.py to selest the best speed for you.
    Car can switch between positve and negative speed (go reverse) without any problem.
    """
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
del rs
del Car
