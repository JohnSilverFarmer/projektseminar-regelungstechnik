import cv2
import numpy as np

ADAPT_THRES_SIZE = 31
ADAPT_THRES_C = 10
BLUR_SIZE = 5
MORPH_ELLIPSE_SIZE = 7
DILATE_ITERS = 2
FILLED_THRES = 0.7


class Circle:
    """ Data structure to represent a circle i.e. center and radius """
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


def detect_circles(src):
    """
    Detects circles in the input image.

    The detection algorithm consists of two main steps:
        1. adaptive threshold & dilate to remove digits
        2. find contours, rejecting non-filled contours

    :param src: The source image. Expected to be a gray scale bird's eye view of the drawing board.
    :return: A tuple (circles, filtered_cicles, filtered_img) where circles are all detected circles,
             fitlered_circles are all detected circles that are filled and filtered_img is the
             intermediate binary image that was used as the bases for contour detection.
    """
    filtered_img = filter_circles(src, ADAPT_THRES_SIZE, ADAPT_THRES_C,
                                  BLUR_SIZE, MORPH_ELLIPSE_SIZE, DILATE_ITERS)

    # now the image should only contain circles, so find them via contours
    circles = contours_circles(filtered_img)
    filterd_circles = filter_filled_circles(filtered_img, circles, FILLED_THRES)
    return circles, filterd_circles, filtered_img


def to_circle(contour, max_radius=300):
    """ Create a circle, i.e. center and radius, from a single
    contour. Optionally filters large contours. """
    r = cv2.boundingRect(contour)[2] // 2
    if r > max_radius:
        return None
    m = cv2.moments(contour)
    x, y = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
    return Circle(x, y, r)


def contours_circles(src):
    """ Finds circles in the source image via cv2.contours().
    Note: The input image should be filter so that it does not
    contain any other regions except circles."""
    if cv2.__version__.startswith('4.'):
        contours = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    else:
        im2, contours, hierarchy = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        circle = to_circle(cnt)
        if circle is not None:
            circles.append(circle)
    return circles


def filter_filled_circles(src, circles, threshold):
    """ Rejects all circles in the binary source imgage that are not filled.
    A circle is filled if: (# black pixels)/(# pixel) > threshold. """
    accepted = []
    for c in circles:
        # get pixels inside the circle
        Y, X = np.ogrid[:src.shape[0], :src.shape[1]]
        mask = (X - c.x) ** 2 + (Y - c.y) ** 2 <= c.r ** 2
        frac_black = float(np.count_nonzero(src[mask] == 0)) / float(np.count_nonzero(mask))

        if frac_black > threshold:
            accepted.append(c)
    return accepted


def filter_circles(src, adap_thres_size, adapt_thres_C, blur_size, morph_ell_size, dilate_iters):
    """ Filter a gray scale bird's eye view in such a way as to remove all digits. """
    thres = cv2.adaptiveThreshold(src, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  adap_thres_size,
                                  adapt_thres_C)

    # binaryzation might have corrupted some circles,
    # restore to remove "spotty" regions
    thres = cv2.medianBlur(thres, blur_size)

    # digits are thinner than points, remove digits via dilation
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ell_size, morph_ell_size))
    thres_dil = cv2.dilate(thres, el, iterations=dilate_iters)

    # dilation shrinks circles, try to restore original size
    # by eroding
    thres_er = cv2.erode(thres_dil, el, iterations=dilate_iters)
    return thres_er
