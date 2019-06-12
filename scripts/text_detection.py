import pytesseract
from pytesseract import Output


class TextBox:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mid_x = int(x + w / 2.0)
        self.mid_y = int(y + h / 2.0)


def detect_boxes(img):
    """
    Detects boxes of text in an image. The actual detection is done via tesseract.
    For more information on parameters and configuration see:
    https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
    """
    # using the legacy engine requires downloading train data from
    # https://github.com/tesseract-ocr/tessdata
    data = pytesseract.image_to_data(img, config='-c tessedit_char_whitelist=0123456789 --psm 6 --oem 0',
                                     output_type=Output.DICT)

    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i]
        if text != u'' and text != u'0':
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append(TextBox(text, x, y, w, h))

    return boxes
