import cv2
import numpy as np
from itertools import combinations


def get_area(img, lower_color, upper_color, debug=False):
    # Threshold the HSV image to get only defined colors
    mask = cv2.inRange(img, lower_color, upper_color)
    n = np.count_nonzero(mask)

    if debug:
        return n, mask
    else:
        return n


def detect_text_color(img, text_boxes, debug=False):
    if not img.ndim == 3:
        raise ValueError('Color detection requires a color image as input.')

    for t in text_boxes:
        # get the textbox subframe as roi
        roi = img[t.y:t.y + t.h, t.x:t.x + t.w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([100, 50, 100])
        upper_blue = np.array([140, 255, 255])

        lower_red_1 = np.array([0, 30, 30])
        upper_red_1 = np.array([20, 255, 255])

        lower_red_2 = np.array([160, 30, 30])
        upper_red_2 = np.array([179, 255, 255])

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([179, 255, 120])

        if debug:
            n_blue, mask_blue = get_area(hsv, lower_blue, upper_blue, debug)

            n_red, mask_red = get_area(hsv, lower_red_1, upper_red_1, debug)
            n_red2, mask_red2 = get_area(hsv, lower_red_2, upper_red_2, debug)

            n_red += n_red2

            n_black, mask_black = get_area(hsv, lower_black, upper_black, debug)
        else:
            n_blue = get_area(hsv, lower_blue, upper_blue, debug)

            n_red = get_area(hsv, lower_red_1, upper_red_1, debug)
            n_red2 = get_area(hsv, lower_red_2, upper_red_2, debug)

            n_red += n_red2

            n_black = get_area(hsv, lower_black, upper_black, debug)

        color_counts = [n_black, n_blue, n_red]

        two_colors_detected = False
        for x, y in combinations(color_counts, 2):
            if x != 0 and y != 0 and ((x < y and float(x)/float(y) > 0.9) or (x > y and float(y)/float(x) > 0.9)):
                two_colors_detected = True

        if two_colors_detected or np.count_nonzero(color_counts) == 0:
            t.color_id = 0
        elif n_black > n_blue and n_black > n_red:
            t.color_id = 1
        elif n_blue > n_red and n_blue > n_black:
            t.color_id = 2
        elif n_red > n_blue and n_red > n_black:
            t.color_id = 3
        else:
            t.color_id = 0
