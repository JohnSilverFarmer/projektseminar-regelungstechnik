import pytesseract
from pytesseract import Output
import cv2
import math
import itertools


class TextBox:
    def __init__(self, text, x, y, w, h, conf):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.set_mid_points()
        self.conf = conf

    def __str__(self):
        return '(Text: ' + self.text + ' x: ' + str(
            self.x) + ' y: ' + str(self.y) + ' w: ' + str(self.w) + ' h: ' + str(self.h) + ' conf: ' + str(
            self.conf) + ')'

    def set_mid_points(self):
        self.mid_x = int(self.x + self.w / 2.0)
        self.mid_y = int(self.y + self.h / 2.0)


def boxes_overlap(b1, b2):
    # Intersection
    x = max(b1[0], b2[0])
    y = max(b1[1], b2[1])
    w = min(b1[0] + b1[2], b2[0] + b2[2]) - x
    h = min(b1[1] + b1[3], b2[1] + b2[3]) - y
    if w <= 0 or h <= 0:
        return False
    else:
        return True


def get_smaller_box(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    a1 = w1 * h1
    a2 = w2 * h2
    if a1 < a2:
        return b1
    else:
        return b2


def get_boxes_distance(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    top_left_1 = (x1, y1)
    top_right_1 = (x1 + w1, y1)
    top_left_2 = (x2, y2)
    top_right_2 = (x2 + w2, y2)
    d1 = math.sqrt((top_left_1[0] - top_right_2[0])**2 + (top_left_1[1] - top_right_2[1])**2)
    d2 = math.sqrt((top_left_2[0] - top_right_1[0])**2 + (top_left_2[1] - top_right_1[1])**2)
    return min([d1, d2])


def detect_single_digit_numbers(img):
    # Retrieving boxes from fount contours
    if cv2.__version__.startswith('4.'):
        contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    else:
        im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes = map(lambda cnt: cv2.boundingRect(cnt), contours)

    # Remove boxes that are too big
    contour_boxes = filter(lambda box: box[2] < 200 and box[3] < 200, contour_boxes)

    # Remove the smaller of two overlapping boxes
    for box, other in itertools.combinations(contour_boxes, 2):
        if boxes_overlap(box, other):
            smaller = get_smaller_box(box, other)
            if smaller in contour_boxes:
                contour_boxes.remove(smaller)
    # Remove all small boxes
    contour_boxes = filter(lambda o: o[2] > 8 and o[3] > 8, contour_boxes)
    # Remove all boxes of multi digit numbers by distance
    to_remove = []
    for box, other in itertools.combinations(contour_boxes, 2):
        h = max(box[3], other[3])
        dist = get_boxes_distance(box, other)
        if dist < h:
            to_remove.append(box)
            to_remove.append(other)

    contour_boxes = filter(lambda box: box not in to_remove, contour_boxes)

    detect_img = cv2.GaussianBlur(img, (5, 5), 2.)
    # Detect text and create TextBoxes
    text_boxes = []
    for id, (x, y, w, h) in enumerate(contour_boxes):
        roi = detect_img[y:y+h, x:x+h]
        data = pytesseract.image_to_data(roi, config='-c tessedit_char_whitelist=123456789 --psm 10 --oem 0',
                                         output_type=Output.DICT)
        text_boxes += get_text_boxes_from_data(data, x, y)

    return text_boxes


def get_text_boxes_from_data(data, x_add=0, y_add=0):
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        conf = float(data['conf'][i])
        text = data['text'][i]
        if conf >= 0 and text != u'0' and text.isdigit():
            (x, y, w, h) = (data['left'][i] + x_add, data['top'][i] + y_add, data['width'][i], data['height'][i])
            box = TextBox(text, x, y, w, h, conf)
            boxes.append(box)

    return boxes


def detect_multi_digit_numbers(img):
    img = cv2.GaussianBlur(img, (5, 5), 1.)
    data = pytesseract.image_to_data(img, config='-c tessedit_char_whitelist=0123456789 --psm 12 --oem 1',
                                     output_type=Output.DICT)
    return get_text_boxes_from_data(data)


def detect_boxes(img, debug):
    """
    Detects boxes of text in an image. The actual detection is done via tesseract.
    For more information on parameters and configuration see:
    https://github.com/tesseract-ocr/tesseract/

    Note: using the legacy engine may require downloading train data from
          https://github.com/tesseract-ocr/tessdata
    """

    # detect text boxes
    text_boxes = detect_single_digit_numbers(img)
    text_boxes += detect_multi_digit_numbers(img)

    # delete duplicates by confidence
    to_remove = []
    for box, other in itertools.combinations(text_boxes, 2):
        if int(box.text) == int(other.text):
            to_remove.append(box) if box.conf < other.conf else to_remove.append(other)

    text_boxes = filter(lambda box: box not in to_remove, text_boxes)

    if debug:
        return text_boxes, img
    else:
        return text_boxes
