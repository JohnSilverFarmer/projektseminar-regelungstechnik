import csv
from circle2text_matching import MalenNachZahlenPunkt
from circle_detection import Circle
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import image
import math

DISTANCE_TEXT_POINT_MM = 10
MIN_DISTANCE_TEXT_TEXT_MM = 10
MM_TO_PIXEL_SCALE = 10
ADD_TO_CENTER_X_MM = 4
ADD_TO_CENTER_Y_MM = 2.5
MIN_ADDITIONAL_DIST_MM = 4

def rotate(length, angle):
    rad = angle * math.pi / 180
    return length * math.cos(rad), length * math.sin(rad)



def main():
    data = csv.reader(open(str(Path('../data/csv-files/PointSetIAT.csv').absolute())), delimiter=',')
    x = []
    y = []
    c = []
    for row in data:
        x.append(str(1.15 * float(row[0])))
        y.append(str(1.15 * float(row[1])))
        c.append(row[2])

    np.savetxt("../data/csv-files/PointSetIATScaled.csv", np.column_stack((x, y, c)), delimiter=",", fmt='%s')


if __name__ == '__main__':
    main()