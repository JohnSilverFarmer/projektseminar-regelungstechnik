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


def backup_detection(img):
    img = cv2.GaussianBlur(img, (5, 5), 1.)
    data = pytesseract.image_to_data(img, config='-l digits --psm 6 --oem 1',
                                     output_type=Output.DICT)
    tmp = get_text_boxes_from_data(data)
    return get_text_boxes_from_data(data)


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
    data = pytesseract.image_to_data(img, config='-l digits --psm 11 --oem 1',
                                     output_type=Output.DICT)
    return get_text_boxes_from_data(data)


def remove_duplicate_texts(text_boxes):
    # delete duplicates by confidence
    to_remove = []
    for box, other in itertools.combinations(text_boxes, 2):
        if int(box.text) == int(other.text):
            to_remove.append(box) if box.conf < other.conf else to_remove.append(other)

    cleaned = filter(lambda box: box not in to_remove, text_boxes)
    return cleaned


def detect_boxes(img, debug):
    """
    Detects boxes of text in an image. The actual detection is done via tesseract.
    For more information on parameters and configuration see:
    https://github.com/tesseract-ocr/tesseract/

    Note: using the legacy engine may require downloading train data from
          https://github.com/tesseract-ocr/tessdata
    """

    # detect text boxes
    thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 30)

    text_boxes = detect_multi_digit_numbers(thres)

    text_boxes = remove_duplicate_texts(text_boxes)
    # now erase detected numbers for backup detection of the rest
    for box in text_boxes:
        cv2.rectangle(thres, (box.x - 2, box.y - 2), (box.x + box.w + 2, box.y + box.h + 2), (255, 255, 255), -1)
    text_boxes += backup_detection(thres)

    text_boxes = remove_duplicate_texts(text_boxes)

    if debug:
        return text_boxes, thres
    else:
        return text_boxes
