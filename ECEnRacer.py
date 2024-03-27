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
from image_processing import (
    get_obstacle,
    get_noodle_not_red,
    depth_straight_contoller,
    depth_to_offset,
)


rs = RealSense("/dev/video2", RS_VGA)  # RS_VGA, RS_720P, or RS_1080P
writer = None

# Use $ ls /dev/tty* to find the serial port connected to Arduino
Car = Arduino("/dev/ttyUSB0", 115200)  # Linux
# Car = Arduino("/dev/tty.usbserial-2140", 115200)    # Mac
Car.pid(1)  # Use PID control
# You can use kd and kp commands to change KP and KD values.  Default values are good.
# loop over frames from Realsense
count = 0
integral_control = 0.0
integral_control_blue = 0.0
angle = 0.0
nood_angle = -10
while True:
    (time, rgb, depth, accel, gyro) = rs.getData()

    if count % 2 == 1:

        """
        Add your code to process rgb, depth, IMU data40
        """

        crop = crop_down(rgb, 120)
        crop = crop_up(crop, 30)
        crop = cv2.resize(crop, (0, 0), fx=0.3, fy=0.5)

        blue_obst = get_noodle_not_red(crop)
        gridx, gridy = 10, 12
        grid_state_machine = make_grid(blue_obst, gridx, gridy, 0.35)
        # cv2.imwrite(f"act{count}.jpg",crop)
        # cv2.imwrite(f"grid{count}.jpg",grid_state_machine*250)
        # cv2.imwrite(f"blueobs{count}.jpg",blue_obst)

        if np.count_nonzero(grid_state_machine[4:, :]) > 0:
            print("AH! NOODS!")
            # crop = get_obstacle(crop)
            # gridx, gridy = 10, 10
            # grid = make_grid(crop, gridx, gridy, 0.35)
            # scalex = crop.shape[1] // gridx
            # scaley = crop.shape[0] // gridy
            # midpoints = grid2midpoints(grid, scalex=scalex, scaley=scaley)
            # angle = find_ave_angle(midpoints)

            angle = nood_angle
            nood_angle += 1
            integral_control = 0.0
            integral_control_blue = 0.0

        elif np.count_nonzero(grid_state_machine[1:6, :]) > 0:
            print("Noods close...")

            # crop = get_obstacle(crop)
            # gridx, gridy = 10, 10
            # grid = make_grid(crop, gridx, gridy, 0.35)
            # scalex = crop.shape[1] // gridx
            # scaley = crop.shape[0] // gridy
            # midpoints = grid2midpoints(grid, scalex=scalex, scaley=scaley)
            # angle = find_ave_angle(midpoints)

            angle, integral_control_blue = \
                depth_straight_contoller(cv2.threshold(blue_obst, 200, 255, cv2.THRESH_BINARY_INV)[1],integral_control_blue,kp=0.05, ki=0.0002)
        else:
            nood_angle = -10
            depth = cv2.resize(depth, (0,0),fx=.3,fy=.3)
            # print("integrator: ", integral_control)
            angle, integral_control = \
                depth_straight_contoller(depth, integral_control,kp=0.04, ki=0.0002)

        """
        Control the Car
        """
        
        Car.zero(1570)  # Set car to go straight.  Change this for your car.
        Car.steer(angle)
        # if count < 40:
        Car.drive(3)
        # else:>
            # Car.drive(2.6)
        # print("Count: ", count, "Angle", end =": ")
        # print(angle)
        # if count % 3 == 0 or True:
        #     cv2.imwrite(f"depth_{count}.jpg", depth)
    else:
        pass
    count += 1
    # if count > 120:
        # break
    """
   	IMPORTANT: Never go full speed. Use CarTest.py to selest the best speed for you.
    Car can switch between positve and negative speed (go reverse) without any problem.
    """
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break
del rs
del Car
