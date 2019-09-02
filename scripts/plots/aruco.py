from pathlib import Path
import cv2
import numpy as np

from scripts.transformation import AR_DICT
import matplotlib.pyplot as plt


IMAGE_FILE = '../../data/test-images/old/test-iat - 1.jpg'
OUT_FILE = Path('~/Desktop/aruco_detect_sample.png').expanduser()

colors_bgr = [tuple(int(s[i:i + len(s) // 3], 16) for i in range(0, len(s), len(s) // 3))[::-1] for s in ['1f77b4',
                                                                                                          'ff7f0e']]


def main():
    img = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
    img = img[520:750, 2950:3250]

    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.detectMarkers(img, AR_DICT, parameters=parameters)

    res_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res_img = cv2.aruco.drawDetectedMarkers(res_img, corners, ids)

    cv2.imwrite(str(OUT_FILE), res_img)


if __name__ == '__main__':
    main()
