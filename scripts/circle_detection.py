import cv2
import numpy as np


class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


# hough circle parameters
BLUR_SIZE = (5, 5)
MIN_DIST = 10
PARAM1 = 100
PARAM2 = 15
MIN_RADIUS = 0

# if the median of a set of pixels is smaller then this value the set of pixels is considered filled
FILLED_THRES = 120


def detect_circles(img, max_radius, reject_empty, debug):
    """
    Detects circles in an image. Uses the hough circle algorithm to find circles.
    Adds an option to test whether a circle is filled or not.
    """
    if img.ndim != 2:
        raise ValueError('Circle detection requires gray scale images.')

    blurred = cv2.GaussianBlur(img, BLUR_SIZE, 3.)

    circles = cv2.HoughCircles(blurred,
                               cv2.HOUGH_GRADIENT, 1.,
                               minDist=MIN_DIST,
                               param1=PARAM1,
                               param2=PARAM2,
                               minRadius=MIN_RADIUS,
                               maxRadius=max_radius)[0]

    detected_circles = []
    for id, (x, y, r) in enumerate(circles.astype(np.int)):
        # get pixels inside the circle
        Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = np.sqrt((X - x) ** 2 + (Y - y) ** 2) <= r

        # if required only accept circles that are mostly filled
        if np.median(img[mask]) < FILLED_THRES or not reject_empty:
            cv2.circle(blurred, (x, y), r*2, (0, 255, 0), 2)
            cv2.putText(blurred, str(id), (x+r, y+r), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
            detected_circles.append(Circle(x, y, r))

    if debug:
        return detected_circles, blurred
    else:
        return detected_circles
