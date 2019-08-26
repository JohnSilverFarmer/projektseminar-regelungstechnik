from scripts.circle_detection import ADAPT_THRES_C, ADAPT_THRES_SIZE, BLUR_SIZE, MORPH_ELLIPSE_SIZE, DILATE_ITERS
import numpy as np
import cv2
import os


INPUT_IMAGE = '../../data/test-images/test-color-8.jpg-bird-eye.png'
OUTPUT_DIR = '/Users/Johannes/Desktop/'
INPUT_ROI = np.index_exp[500:1000, 100:700]


def main():
    src = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    src = src[INPUT_ROI]

    thres = cv2.adaptiveThreshold(src, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  ADAPT_THRES_SIZE,
                                  ADAPT_THRES_C)
    thres = cv2.medianBlur(thres, BLUR_SIZE)

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'circle_detect_thres_and_filter.png'), thres)

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_ELLIPSE_SIZE, MORPH_ELLIPSE_SIZE))
    thres_dil = cv2.dilate(thres, el, iterations=DILATE_ITERS)

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'circle_detect_erode.png'), thres_dil)

    thres_er = cv2.erode(thres_dil, el, iterations=DILATE_ITERS)

    cv2.imwrite(os.path.join(OUTPUT_DIR, 'circle_detect_dilate.png'), thres_er)


if __name__ == '__main__':
    main()
