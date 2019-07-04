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
        self.mid_x = int(x + w / 2.0)
        self.mid_y = int(y + h / 2.0)
        self.conf = conf

    def __str__(self):
        return '(Text: ' + self.text + ' x: ' + str(
            self.x) + ' y: ' + str(self.y) + ' w: ' + str(self.w) + ' h: ' + str(self.h) + ' conf: ' + str(
            self.conf) + ')'


def boxes_overlap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    if (x1 in range(x2, x2 + w2) and y1 in range(y2, y2 + h2)) or (
            x2 in range(x1, x1 + w1) and y2 in range(y1, y1 + h1)):
        return True
    else:
        return False


def get_smaller_box(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    a1 = w1 * h1
    a2 = w2 * h2
    if a1 < a2:
        return b1
    else:
        return b2


def detect_single_digit_numbers(img):
    # Retrieving boxes from fount contours
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_boxes = map(lambda cnt: cv2.boundingRect(cnt), contours)

    # Remove boxes that are too big
    contour_boxes = filter(lambda box: box[2] < 200 and box[3] < 200, contour_boxes)

    # Remove the smaller of two overlapping boxes
    for box, other in itertools.combinations(contour_boxes, 2):
        if boxes_overlap(box, other):
            contour_boxes.remove(get_smaller_box(box, other))

    # Remove near-quadratic boxes (should be the circles)
    contour_boxes = filter(lambda o: not (o[2] in range(o[3] - 2, o[3] + 2)), contour_boxes)

    # Remove all boxes of multi digit numbers by distance
    to_remove = []
    for box, other in itertools.combinations(contour_boxes, 2):
        x, y, w, h = box
        x_o, y_o, w_o, h_o = other
        dist_1 = math.sqrt((x + w - x_o) ** 2 + (y - y_o) ** 2)
        dist_2 = math.sqrt((x_o + w_o - x) ** 2 + (y_o - y) ** 2)
        if dist_1 < h or dist_2 < h:
            to_remove.append(box)
            to_remove.append(other)

    contour_boxes = filter(lambda box: box not in to_remove, contour_boxes)

    # Detect text and create TextBoxes
    text_boxes = []
    for (x, y, w, h) in contour_boxes:
        roi = img[y:y+h, x:x+h]
        data = pytesseract.image_to_data(roi, config='-c tessedit_char_whitelist=0123456789 --psm 10 --oem 0',
                                         output_type=Output.DICT)

        text_boxes += get_text_boxes_from_data(data, x, y)

    return text_boxes


def get_text_boxes_from_data(data, x_add=0, y_add=0):
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        conf = float(data['conf'][i])
        text = data['text'][i]
        if conf > 30 and text != u'0' and text.isdigit():
            (x, y, w, h) = (data['left'][i] + x_add, data['top'][i] + y_add, data['width'][i], data['height'][i])
            box = TextBox(text, x, y, w, h, conf)
            boxes.append(box)

    return boxes


def detect_multi_digit_numbers(img):
    data = pytesseract.image_to_data(img, config='-c tessedit_char_whitelist=0123456789 --psm 11 --oem 1',
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
    # pre-processing to convert the image to binary using an adaptive threshold
    thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 30)
    thres = cv2.medianBlur(thres, 5)

    # detect text boxes
    text_boxes = detect_single_digit_numbers(thres)
    text_boxes += detect_multi_digit_numbers(thres)

    # delete duplicates by confidence
    to_remove = []
    for box, other in itertools.combinations(text_boxes, 2):
        if int(box.text) == int(other.text):
            to_remove.append(box) if box.conf < other.conf else to_remove.append(other)

    text_boxes = filter(lambda box: box not in to_remove, text_boxes)

    if debug:
        return text_boxes, thres
    else:
        return text_boxes
