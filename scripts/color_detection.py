import cv2
import numpy as np

def check_for_color(img, lower_color, upper_color):
    color_exists = False

    # Threshold the HSV image to get only defined colors
    mask = cv2.inRange(img, lower_color, upper_color)
    color_contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if len(color_contours) > 0:
        color_exists = True

    return color_exists

def detect_text_color(img, mnz_point):
    if not img.ndim == 3:
        raise ValueError('Color detection requires a color image as input.')

    # get the textbox subframe as roi
    t = mnz_point.textBox
    roi = img[t.y:t.y+t.h, t.x:t.x+t.w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100, 10, 10])
    upper_blue = np.array([140, 255, 255])

    lower_red_1 = np.array([0, 50, 50])
    upper_red_1 = np.array([20, 255, 255])

    lower_red_2 = np.array([160, 50, 50])
    upper_red_2 = np.array([179, 255, 255])

    if check_for_color(hsv, lower_blue, upper_blue):
        mnz_point.color_id = 1
    elif check_for_color(hsv, lower_red_1, upper_red_1) or check_for_color(hsv, lower_red_2, upper_red_2):
        mnz_point.color_id = 2
    else:
        mnz_point.color_id = 0
