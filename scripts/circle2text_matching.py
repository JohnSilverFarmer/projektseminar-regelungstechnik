import numpy as np


class MalenNachZahlenPunkt:

    def __init__(self, num_id, color_id, x, y):
        self.num_id = num_id
        self.color_id = color_id
        self.x = x
        self.y = y

    def __str__(self):
        return '(Number: ' + str(self.num_id) + ' Color: ' + str(self.color_id) + ' x: ' + str(self.x) + ' y: ' + str(
            self.y)


def match(circles, text_boxes):
    """
    Match detected circles and to corresponding text boxes.
    """
    recognized_points = []
    for t in text_boxes:
        min_distance = None
        nearest_circle = None
        for c in circles:
            cur_distance = np.sqrt((t.mid_x - c.x) ** 2 + (t.mid_y - c.y) ** 2)
            if min_distance is None or cur_distance < min_distance:
                min_distance = cur_distance
                nearest_circle = c
        recognized_points.append(MalenNachZahlenPunkt(int(t.text), 1, nearest_circle.x, nearest_circle.y))

    recognized_points.sort(key=lambda p: p.num_id)
    return recognized_points
