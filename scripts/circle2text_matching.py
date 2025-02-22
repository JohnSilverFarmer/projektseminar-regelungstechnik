import numpy as np


class MalenNachZahlenPunkt:
    def __init__(self, num_id, circle, text_box):
        self.num_id = num_id
        self.circle = circle
        self.textBox = text_box

    def __str__(self):
        return '(Number: ' + str(self.num_id) + ' x: ' + str(
            self.circle.x) + ' y: ' + str(self.circle.y) + ')'


def match(circles, text_boxes):
    """
    Match detected circles to corresponding text boxes. For each text box
    the corresponding circle is the circle who's center is closest to the
    text box center.
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
        recognized_points.append(MalenNachZahlenPunkt(int(t.text), nearest_circle, t))

    recognized_points.sort(key=lambda p: p.num_id)
    return recognized_points
