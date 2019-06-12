import pytesseract
from pytesseract import Output


class TextBox:
    def __init__(self, text, x, y, w, h):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def detect_boxes(img):
    # https://github.com/tesseract-ocr/tesseract/wiki/ImproveQuality
    #  4     Assume a single column of text of variable sizes.
    #  6     Assume a single uniform block of text.
    #  11    Sparse text. Find as much text as possible in no particular order.
    data = pytesseract.image_to_data(img, config='-c tessedit_char_whitelist=0123456789 --psm 6 --oem 0',
                                     output_type=Output.DICT)
    h, w, _ = img.shape

    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i]
        if text != u'' and text != u'0':
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append(TextBox(text, x, y, w, h))
