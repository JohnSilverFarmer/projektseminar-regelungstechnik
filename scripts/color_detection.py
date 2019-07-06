import cv2
import numpy as np


def get_area(img, lower_color, upper_color):
    # Threshold the HSV image to get only defined colors
    mask = cv2.inRange(img, lower_color, upper_color)
    n = np.count_nonzero(mask)

    return n


def detect_text_color(img, mnz_points):
    if not img.ndim == 3:
        raise ValueError('Color detection requires a color image as input.')

    for pt in mnz_points:
        # get the textbox subframe as roi
        t = pt.textBox
        roi = img[t.y:t.y + t.h, t.x:t.x + t.w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_green = np.array([40, 10, 10])
        upper_green = np.array([79, 255, 255])

        lower_red_1 = np.array([0, 50, 50])
        upper_red_1 = np.array([20, 255, 255])

        lower_red_2 = np.array([160, 50, 50])
        upper_red_2 = np.array([179, 255, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([179, 140, 140])

        n_green = get_area(hsv, lower_green, upper_green)

        n_red = get_area(hsv, lower_red_1, upper_red_1)
        n_red += get_area(hsv, lower_red_2, upper_red_2)

        n_black = get_area(hsv, lower_black, upper_black)

        if n_black > n_green and n_black > n_red:
            pt.color_id = 1
        elif n_green > n_red and n_green > n_black:
            pt.color_id = 1
        else:
            pt.color_id = 2

