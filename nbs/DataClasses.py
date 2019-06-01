
class TextData:

    def __init__(self, text, x, y, w, h):
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.mid_x = int(x + w / 2.0)
        self.mid_y = int(y + h / 2.0)


class CircleData:

    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r


class MalenNachZahlenPunkt:

    def __init__(self, num_id, color_id, x, y):
        self.num_id = num_id
        self.color_id = color_id
        self.x = x
        self.y = y

    def __str__(self):
        return '(Number: ' + str(self.num_id) + ' Color: ' + str(self.color_id) + ' x: ' + str(self.x) + ' y: ' + str(self.y)