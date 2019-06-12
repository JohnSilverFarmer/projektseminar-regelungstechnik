import cv2
import numpy as np


class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


def detect_circles(img, max_radius, reject_empty):
    circles = cv2.HoughCircles(cv2.GaussianBlur(img, (5, 5), 2.), cv2.HOUGH_GRADIENT, 1.,
                               minDist=10, param1=100, param2=15,
                               minRadius=0, maxRadius=max_radius)[0]

    detected_circles = []
    for x, y, r in circles.astype(np.int):
        Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) <= r

        # only accept circles that are mostly filled with black
        if np.median(img[mask]) < 120 or not reject_empty:
            detected_circles.append(Circle(x, y, r))

    return detected_circles
