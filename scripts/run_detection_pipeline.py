import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

from transformation import detect_markers_and_compute
from circle_detection import detect_circles
from text_detection import detect_boxes
from circle2text_matching import match
from color_detection import detect_text_color
from detection_editor import DetectionEditor, is_correct


BORDER_FRACTION_TO_CUT = 0.01

W_WORLD, H_WORLD = 420, 297  # size of area marked by inner points of aruco markers in mm
WORLD2IMG_SCALE = 7.5  # scale factor for transforming cm to pixels in the birds eye image
H_IMG, W_IMG = int(H_WORLD * WORLD2IMG_SCALE), int(W_WORLD * WORLD2IMG_SCALE)
H_OFFSET = int(H_IMG * BORDER_FRACTION_TO_CUT)
W_OFFSET = int(W_IMG * BORDER_FRACTION_TO_CUT)


def draw_result(img, circles, mnz_points):
    _img = img.copy()

    if not _img.ndim == 3:
        raise ValueError('Result should be drawn into a color image.')

    for c in circles:
        cv2.circle(_img, (c.x, c.y), c.r, (0, 255, 0), 4)

    for idx, mnz_point in enumerate(mnz_points):
        color = None
        if mnz_point.textBox.color_id == 1:
            color = (0, 0, 0)
        elif mnz_point.textBox.color_id == 2:
            color = (255, 255, 0)
        elif mnz_point.textBox.color_id == 3:
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


def cut_edges(images):
    return map(lambda img: img[H_OFFSET:H_IMG - H_OFFSET, W_OFFSET:W_IMG - W_OFFSET], images)


def apply_offsets(circles, mnz_points):
    for c in circles:
        c.x = c.x + W_OFFSET
        c.y = c.y + H_OFFSET
    for pt in mnz_points:
        pt.textBox.x = pt.textBox.x + W_OFFSET
        pt.textBox.y = pt.textBox.y + H_OFFSET
        pt.textBox.set_mid_points()


def transform_coord_to_rw(x, y):
    x_world_m = float(x) / (WORLD2IMG_SCALE * 1000.) - 0.020
    y_world_m = (H_IMG - float(y))/(WORLD2IMG_SCALE * 1000.) - 0.0185
    return x_world_m, y_world_m


def main(img_file, output_file, debug):
    # read the input image
    img = cv2.imread(img_file)
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.99)
    img_wb = wb.balanceWhite(img)

    m, debug_img_marker = detect_markers_and_compute(img_gs, (H_IMG, W_IMG), debug_image=True)

    # transform the image into a top down perspective
    img_warped = cv2.warpPerspective(img, m, (W_IMG, H_IMG))
    img_warped_gs = cv2.warpPerspective(img_gs, m, (W_IMG, H_IMG))
    img_warped_wb = cv2.warpPerspective(img_wb, m, (W_IMG, H_IMG))

    # cut edges
    img_cutted_gs, img_cutted_wb = cut_edges([img_warped_gs, img_warped_wb])

    # detect circles and text
    img_text = img_cutted_gs
    _, circles, debug_img_circle = detect_circles(img_cutted_gs)
    for c in circles:
        cv2.circle(img_text, (c.x, c.y), int(c.r * 2), (255, 255, 255), -1)

    text_boxes, debug_img_text = detect_boxes(img_text, debug)

    # detect the text color to determine line styles
    detect_text_color(img_cutted_wb, text_boxes)

    if not is_correct(text_boxes, circles):
        editor = DetectionEditor(img_cutted_wb, circles, text_boxes)
        editor.show()

    # find corresponding points and text
    mnz_points = match(circles, text_boxes)

    apply_offsets(circles, mnz_points)

    if debug:
        result_img = draw_result(img_warped, circles, mnz_points)
        imshow([debug_img_marker, result_img])
        plt.savefig(str(Path(output_file).parent/'../debug-images/{}'.format(Path(img_file).name)), dpi=400)

    mnz_points.sort(key=lambda pt: pt.num_id)
    with open(output_file, mode='w') as out:
        out_writer = csv.writer(out, delimiter=',')
        for pt in mnz_points:
            x, y = transform_coord_to_rw(pt.circle.x, pt.circle.y)
            out_writer.writerow([x, y, pt.textBox.color_id - 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect points and corresponding text labels in an image.')
    parser.add_argument('--image', type=str, help='Input image file.')
    parser.add_argument('--outFile', type=str, help='Output file name.')
    parser.add_argument('--debug', help='Show debug information after execution.', action='store_true')

    args = parser.parse_args()
    main(args.image, args.outFile, args.debug)
