import csv
from circle2text_matching import MalenNachZahlenPunkt
from text_detection import TextBox
from circle_detection import Circle
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import image
import math


CIRCLE_RADIUS_MM = 5
FONT_SCALE = 6
FONT_THICKNESS = 5

DISTANCE_TEXT_POINT_MM = 10
MIN_DISTANCE_TEXT_TEXT_MM = 13
MM_TO_PIXEL_SCALE = 10 * 4961./4200
ADD_TO_CENTER_X_MM = 5
ADD_TO_CENTER_Y_MM = 3
MIN_ADDITIONAL_DIST_MM = 4


def rotate(length, angle):
    rad = angle * math.pi / 180
    return length * math.cos(rad), length * math.sin(rad)


def main():
    data = csv.reader(open(str(Path('../data/csv-files/PointSetIAT.csv').absolute())), delimiter=',')
    mnz_points = []
    for idx, row in enumerate(data):
        color_id = int(row[2])
        x_m = row[0]
        y_m = row[1]
        x_mm = float(row[0]) * 1000 + 20
        y_mm = 297 - (float(row[1]) * 1000 + 18.5)
        r_mm = CIRCLE_RADIUS_MM
        mnz_points.append(MalenNachZahlenPunkt(idx + 1, Circle(x_mm, y_mm, r_mm), TextBox(str(idx + 1), color_id=color_id)))

    colors = [(0, 0, 0), (0, 255, 255), (255, 0, 0)]
    text_points = []
    # white blank image
    img = 255 * np.ones(shape=[297 * MM_TO_PIXEL_SCALE, 420 * MM_TO_PIXEL_SCALE, 3], dtype=np.uint8)
    failed = False
    for startPos in range(30, 390, 1):
        for mnz_pt in mnz_points:
            if failed:
                break
            x = int(mnz_pt.circle.x * MM_TO_PIXEL_SCALE)
            y = int(mnz_pt.circle.y * MM_TO_PIXEL_SCALE)
            r = int(mnz_pt.circle.r * MM_TO_PIXEL_SCALE)
            cv2.circle(img, (x, y), r, (0, 0, 0), -1)
            for angle in range(startPos, startPos + 360, 1):
                failed = False
                # Try to put text where no text is yet
                x_add, y_add = rotate(DISTANCE_TEXT_POINT_MM * MM_TO_PIXEL_SCALE, angle)
                x_t_center = x - x_add
                y_t_center = y - y_add
                x_t_lb = x_t_center - ADD_TO_CENTER_X_MM * MM_TO_PIXEL_SCALE
                y_t_lb = y_t_center + ADD_TO_CENTER_Y_MM * MM_TO_PIXEL_SCALE
                for pt in text_points:
                    dist_to_next_text = math.sqrt((x_t_center - pt[0]) ** 2 + (y_t_center - pt[1]) ** 2)
                    if dist_to_next_text < MIN_DISTANCE_TEXT_TEXT_MM * MM_TO_PIXEL_SCALE:
                        failed = True
                        break
                for other_pt in mnz_points:
                    if not (other_pt.circle.x == mnz_pt.circle.x and other_pt.circle.y == mnz_pt.circle.y):
                        x_other = other_pt.circle.x * MM_TO_PIXEL_SCALE
                        y_other = other_pt.circle.y * MM_TO_PIXEL_SCALE
                        x_dist = x_other - x_t_lb
                        y_dist = y_other - y_t_lb
                        dist_to_next_dot = math.sqrt(x_dist ** 2 + y_dist ** 2)
                        if dist_to_next_dot < (DISTANCE_TEXT_POINT_MM + MIN_ADDITIONAL_DIST_MM) * MM_TO_PIXEL_SCALE:
                            failed = True
                            break
                if not failed:
                    text_points.append((x_t_center, y_t_center))
                    cv2.putText(img, str(mnz_pt.num_id), (int(x_t_lb), int(y_t_lb)), cv2.FONT_HERSHEY_PLAIN, FONT_SCALE,
                                colors[int(mnz_pt.textBox.color_id)], FONT_THICKNESS)
                    # cv2.circle(img, (int(x_t_center), int(y_t_center)), 5, (0, 0, 0), -1)
                    break

        if not failed:
            break

    if failed:
        print('Failed to create, the numbers are packed too dense!')
    else:
        image.imsave('../data/mnz-vorlagen/iat-cr-{}.pdf'.format(CIRCLE_RADIUS_MM), img, dpi=300)
        image.imsave('../data/mnz-vorlagen/iat-cr-{}.jpg'.format(CIRCLE_RADIUS_MM), img, dpi=300)


if __name__ == '__main__':
    main()
