import pytesseract
from pytesseract import Output
import cv2


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

    def __eq__(self, other):
        """
        A TextBox equals another if they have the same text inside
        """
        if self.text == other.text:
            return True
        else:
            return False


def detect_boxes(img):
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

    data = pytesseract.image_to_data(thres, config='-c tessedit_char_whitelist=0123456789 --psm 6 --oem 0',
                                     output_type=Output.DICT)

    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        conf = float(data['conf'][i])
        text = data['text'][i]
        if conf > 30 and text != u'0' and text.isdigit():
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            box = TextBox(text, x, y, w, h, conf)
            if box in boxes:
                # Handle duplicates, take the box with the higher confidence
                for b2 in boxes:
                    if box == b2 and box.conf > b2.conf:
                        boxes.remove(b2)
                        boxes.append(box)
            else:
                # Just append
                boxes.append(box)

    return boxes
