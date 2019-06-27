import cv2
import numpy as np
import collections


def get_pixel_count(img, lower_color, upper_color):
    color_exists = False

    # Threshold the HSV image to get only defined colors
    mask = cv2.inRange(img, lower_color, upper_color)
    n = np.count_nonzero(mask)

    return n


def detect_text_color(img, mnz_point):
    if not img.ndim == 3:
        raise ValueError('Color detection requires a color image as input.')

    # get the textbox subframe as roi
    t = mnz_point.textBox
    roi = img[t.y:t.y + t.h, t.x:t.x + t.w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100, 10, 10])
    upper_blue = np.array([140, 255, 255])

    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([20, 255, 255])

    lower_red_2 = np.array([160, 50, 50])
    upper_red_2 = np.array([179, 255, 255])

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 70])


    n_blue = get_pixel_count(hsv, lower_blue, upper_blue)

    n_red = get_pixel_count(hsv, lower_red_1, upper_red_1)
    n_red += get_pixel_count(hsv, lower_red_2, upper_red_2)

    n_black = get_pixel_count(hsv, lower_black, upper_black)

    if n_blue == 0 and n_red == 0:
        mnz_point.color_id = 0
    elif n_blue > n_red:
        mnz_point.color_id = 1
    else:
        mnz_point.color_id = 2
