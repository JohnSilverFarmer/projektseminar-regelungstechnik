import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

from transformation import detect_markers_and_compute
from circle_detection import detect_circles
from text_detection import detect_boxes
from circle2text_matching import match
from color_detection import detect_text_color


def draw_result(img, circles, text_boxes, mnz_points):
    _img = img.copy()

    if not _img.ndim == 3:
        raise ValueError('Result should be drawn into a color image.')

    for c in circles:
        cv2.circle(_img, (c.x, c.y), c.r, (0, 255, 0), 4)

    for idx, mnz_point in enumerate(mnz_points):
        color = None
        if mnz_point.color_id == 0:
            color = (0, 0, 0)
        elif mnz_point.color_id == 1:
            color = (255, 0, 0)
        elif mnz_point.color_id == 2:
            color = (0, 0, 255)

        t = mnz_point.textBox
        cv2.putText(_img, t.text, (t.x + t.w, t.y), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
        cv2.rectangle(_img, (t.x, t.y), (t.x + t.w, t.y + t.h), color, 4)

        if idx != len(mnz_points) - 1:
            p1 = mnz_point
            p2 = mnz_points[idx + 1]
            if not (color == (0, 0, 0)):
                cv2.line(_img, (p1.circle.x, p1.circle.y), (p2.circle.x, p2.circle.y), color, 2)

    return _img


def imshow(imgs, figsize=(12.5, 10), **kwargs):
    """
    Utility function for dispalying an array of images.
    """
    if not isinstance(imgs, list): imgs = [imgs]
    cols = int(np.ceil(np.sqrt(len(imgs))))
    rows = cols - 1 if (cols * cols - len(imgs) == cols) else cols
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    for img, ax in zip(imgs, axs.flatten()):
        cm, sl = ('gray', np.s_[:, :]) if img.ndim == 2 else (None, np.s_[..., ::-1])
        ax.imshow(img[sl], cmap=cm, **kwargs)
    for ax in axs.flatten(): ax.axis('off')
    plt.tight_layout()


def main(args):
    # read the input image
    img = cv2.imread(args.image)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w_world, h_world = 2 * 21.0, 29.7  # size of area marked by inner points of aruco markers in cm
    world2img_scale = 50.  # scale factor for transforming cm to pixels in the birds eye image
    h_img, w_img = int(h_world * world2img_scale), int(w_world * world2img_scale)

    m, debug_img_marker = detect_markers_and_compute(img_gs, (h_img, w_img), debug_image=True)

    # transform the image into a top down perspective
    img_warped = cv2.warpPerspective(img, m, (w_img, h_img))
    img_warped_gs = cv2.warpPerspective(img_gs, m, (w_img, h_img))

    # detect circles and text
    circles = detect_circles(img_warped_gs, w_img // 200, reject_empty=True)
    text_boxes = detect_boxes(img_warped_gs)

    # find corresponding points and text
    mnz_points = match(circles, text_boxes)

    # detect the text color to determine line styles
    for mnz_point in mnz_points:
        detect_text_color(img_warped, mnz_point)

    if args.debug:
        result_img = draw_result(img_warped, circles, text_boxes, mnz_points)
        imshow([debug_img_marker, result_img])
        plt.show()

    # TODO: export detections as a csv file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect points and corresponding text labels in an image.')
    parser.add_argument('--image', type=str, help='Input image file.')
    parser.add_argument('--outFile', type=str, help='Output file name.')
    parser.add_argument('--debug', help='Show debug information after execution.', action='store_true')

    args = parser.parse_args()
    main(args)
