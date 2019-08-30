import matplotlib.pyplot as plt
import numpy as np
import cv2

from pathlib import Path


# This file plots a few examples for the basic filter
# operations introduced in the project report.


def _add_noise(image, p=0.07):
    image += np.random.choice([0, 255], image.shape, p=[1 - p, p])
    return np.clip(image, 0, 255)


def _generate_digit_example(digit=8, with_noise=True):
    """ Generate an image containing a single digit to demonstrate filters- """
    IMG_SIZE = (100, 100)
    FG_COLOR = (255, 255, 255)

    image = np.zeros(IMG_SIZE)
    cv2.putText(image, str(digit), (15, IMG_SIZE[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 3.5,
                FG_COLOR, 10, lineType=cv2.LINE_AA)

    if with_noise:
        image = _add_noise(image, p=0.07)

    return image.astype(np.uint8)


def _generate_square_example(with_noise=True):
    """ Generate an image containing only a single square. """
    IMG_SIZE = (2000, 2000)

    image = np.zeros(IMG_SIZE)
    rect_size = (1200, 1200)

    m = (IMG_SIZE[0]//2, IMG_SIZE[1]//2)
    image[m[0]-rect_size[0]//2:m[0]+rect_size[0]//2, m[1]-rect_size[1]//2:m[1]+rect_size[1]//2] = 255

    if with_noise:
        image = _add_noise(image, p=0.01)

    return image.astype(np.uint8)


def main():
    out_dir = Path('~/Desktop/filter_examples').expanduser()
    out_dir.mkdir(exist_ok=True)

    # =====================
    # Morphological Filters
    # =====================
    rect_image = _generate_square_example(with_noise=False)

    struc_size = (300, 300)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, struc_size)

    rect_dil = cv2.dilate(rect_image, el, iterations=1)
    rect_dil[rect_dil == 255] = 75
    rect_dil[rect_image == 255] = 255

    # draw the structuring element
    str_rgb = (255, 127, 14)
    rect_dil = cv2.cvtColor(rect_dil, cv2.COLOR_GRAY2BGR)
    m = rect_image.shape[0]//2, rect_image.shape[1]//2
    org = (m[0] - 600, m[1] - 600)
    cv2.circle(rect_dil, org, struc_size[0]//2, str_rgb[::-1], -1, cv2.LINE_AA)

    rect_ero = cv2.erode(rect_image, el, iterations=1)
    rect_res = rect_image.copy()
    rect_res[rect_ero == 255] = 75

    # draw the structuring element
    rect_res = cv2.cvtColor(rect_res, cv2.COLOR_GRAY2BGR)
    org = (m[0] - 600 + struc_size[0]//2, m[1] - 600 + struc_size[1]//2)
    cv2.circle(rect_res, org, struc_size[0] // 2, str_rgb[::-1], -1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir.joinpath('morph_orig.png')), rect_image)
    cv2.imwrite(str(out_dir.joinpath('morph_erosion.png')), rect_res)
    cv2.imwrite(str(out_dir.joinpath('morph_dilate.png')), rect_dil)

    # ===================
    # Gaussian and Median
    # ===================
    eight_img = _generate_digit_example()
    eight_gauss = cv2.GaussianBlur(eight_img, (3, 3), sigmaX=3)
    eight_median = cv2.medianBlur(eight_img, 3)

    cv2.imwrite(str(out_dir.joinpath('basic_orig.png')), eight_img)
    cv2.imwrite(str(out_dir.joinpath('basic_gauss.png')), eight_gauss)
    cv2.imwrite(str(out_dir.joinpath('basic_median.png')), eight_median)


if __name__ == '__main__':
    main()
