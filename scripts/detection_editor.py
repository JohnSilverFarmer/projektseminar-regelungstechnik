from circle_detection import Circle
from text_detection import TextBox

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets as wdg
from matplotlib import gridspec

_draw_colors = {0: [255, 0, 255],
                1: [0, 0, 0],
                2: [255, 0, 0],
                3: [0, 0, 255]}


def draw_text(src, text_box):
    draw_color = _draw_colors[text_box.color_id]

    # detected number
    loc = (text_box.x + text_box.w, text_box.y)
    cv2.putText(src, text_box.text, loc, cv2.FONT_HERSHEY_SIMPLEX, 2, draw_color, 2, cv2.LINE_AA)

    # rectangle around detection
    loc = (text_box.x, text_box.y)
    size = (text_box.x + text_box.w, text_box.y + text_box.h)
    cv2.rectangle(src, loc, size, draw_color, 4)


def draw_circle(src, pnt):
    # draw the point
    cv2.circle(src, (pnt.x, pnt.y), pnt.r, (0, 255, 0), 10)


def is_correct(text_boxes, circles):
    detected_numbers = list(map(lambda tb: int(tb.text), text_boxes))
    colors = list(map(lambda tb: tb.color_id, text_boxes))
    color_not_detected = any(c_id == 0 for c_id in colors)
    if max(detected_numbers) != len(detected_numbers) or len(text_boxes) < len(circles) or \
            color_not_detected or len(circles) == 0:
        result = False
    else:
        result = True

    return result


def _compute_box_circle_dims(boxes, circles):
    if len(boxes) > 0:
        box_heigh = int(np.median([b.h for b in boxes]))
        box_width = int(np.median([b.w // len(b.text) for b in boxes]))
    else:
        box_heigh, box_width = 30, 20

    circle_dimension = int(np.median([c.r for c in circles])) if len(circles) != 0 else 4
    return box_heigh, box_width, circle_dimension


class DetectionEditor(object):
    """ A GUI that enables editing detected points and text on an image. You initialize an editor object
    with your detected circles & text boxes and present it to the user via its show() method.

    After the user is done editing the did_edit property of the object is set to true if any actual edits
    were made. The edited circles and text boxes can be accessed via the circles and boxes properties.
    """

    def __init__(self, base_image, circles, boxes):
        self.base_image, self.circles, self.boxes = base_image, circles, boxes
        self.content_image, self.fig, self.ax = None, None, None
        self.did_edit = False
        self._current_edit_mode = None
        self._digit_buffer = ''
        self.add_pnt_button, self.add_num_button, self.delete_button, self.change_color_button = None, None, None, None
        self.exit_button = None

        # compute dimensions for new boxes/circles based on median
        # size of already existing boxes
        self.box_height, self.box_width, self.circle_rad = _compute_box_circle_dims(boxes, circles)

    def _update_content(self):
        updated_image = self.base_image.copy()

        # redraw detections
        for box in self.boxes:
            draw_text(updated_image, box)
        for circle in self.circles:
            draw_circle(updated_image, circle)

        # make sure update is visible
        self.content_image.set_data(updated_image[..., ::-1])
        self.fig.canvas.draw_idle()

    def _disp_mode(self):
        disp_string = ''
        if self._current_edit_mode == 'del':
            disp_string = 'Delete'
        elif self._current_edit_mode == 'num':
            disp_string = 'Add Number' + ' ({})'.format(self._digit_buffer)
        elif self._current_edit_mode == 'point':
            disp_string = 'Add Point'
        elif self._current_edit_mode == 'color':
            disp_string = 'Color' + ' ({})'.format(self._digit_buffer)
        self.fig.suptitle(disp_string, fontsize=16)
        self.fig.canvas.draw_idle()

    def _initialize(self):
        # show detections specified at the beginning
        self.content_image = self.ax.imshow(np.full_like(self.base_image, 255))
        plt.tight_layout()
        self._update_content()

        self.fig.subplots_adjust(bottom=0.15, top=0.92)
        gs = gridspec.GridSpec(1, 5)
        gs.update(left=0.2, right=0.8, bottom=0.05, top=0.1, hspace=0.0)
        axes = [self.fig.add_subplot(gs[i, j]) for i, j in [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]]

        # create editor buttons
        self.add_pnt_button = wdg.Button(axes[0], 'Add Point')
        self.add_pnt_button.on_clicked(self._on_add_point_click)

        self.add_num_button = wdg.Button(axes[1], 'Add Number')
        self.add_num_button.on_clicked(self._on_add_num_click)

        self.delete_button = wdg.Button(axes[2], 'Delete')
        self.delete_button.on_clicked(self._on_delete_click)

        self.change_color_button = wdg.Button(axes[3], 'Change Color')
        self.change_color_button.on_clicked(self._on_change_color_click)

        self.exit_button = wdg.Button(axes[4], 'Save & Exit')
        self.exit_button.on_clicked(self._on_exit_click)

        cid = self.fig.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        if (self._current_edit_mode in ['num', 'color']) and event.key.isdigit():
            self._digit_buffer += event.key
            self._disp_mode()

    def _find_closest(self, loc):
        """ Find the closest detection to a coordinate pair, return the distance
        and whether its a circle or text. """

        def min_dist(a, x):
            if len(a) == 0:
                return None
            dsts = [np.sqrt((el.x - x[0]) ** 2 + (el.y - x[1]) ** 2) for el in a]
            return np.min(dsts), np.argmin(dsts)

        min_pnts, min_nums = min_dist(self.circles, loc), min_dist(self.boxes, loc)

        if min_pnts is None and min_nums is None:
            return None
        if min_pnts is None:
            return self.boxes[min_nums[1]], min_nums[0], True
        elif min_nums is None:
            return self.circles[min_pnts[1]], min_pnts[0], False
        else:
            min_is_num = min_nums[0] < min_pnts[0]
            min_dist = min_nums[0] if min_is_num else min_pnts[0]
            min_idx = min_nums[1] if min_is_num else min_pnts[1]
            content = self.boxes if min_is_num else self.circles
            return content[min_idx], min_dist, min_is_num

    def _on_canvas_click(self, event):
        # don't do anything if in zoom/pan mode
        if plt.get_current_fig_manager().toolbar.mode != '':
            return
        click_loc = int(event.xdata), int(event.ydata)
        did_make_a_change = False

        if self._current_edit_mode == 'del':
            # check if user clicked on a detection and if so delete it
            closest = self._find_closest(click_loc)
            if closest is not None:
                elem, dist, is_num = closest

                if dist <= 15:
                    if is_num:
                        self.boxes.remove(elem)
                    else:
                        self.circles.remove(elem)
                did_make_a_change = True
        elif self._current_edit_mode == 'num':
            # place the entered text on the image
            text = self._digit_buffer
            if text != '':
                width = len(text) * self.box_width
                height = self.box_height
                box = TextBox(x=click_loc[0] - width // 2, y=click_loc[1] - height // 2, text=text,
                              w=width, h=height, color_id=0)
                self.boxes.append(box)

                # reset digit buffer
                self._digit_buffer = ''
                self._disp_mode()

                did_make_a_change = True
        elif self._current_edit_mode == 'point':
            # add a point at the clicked location
            self.circles.append(Circle(x=click_loc[0], y=click_loc[1], r=self.circle_rad))
            did_make_a_change = True
        elif self._current_edit_mode == 'color':
            closest = self._find_closest(click_loc)
            if closest is not None:
                elem, dist, is_num = closest
                # if key press location is close to number change it's color
                if dist <= 10 and is_num and self._digit_buffer != '':
                    elem.color_id = int(self._digit_buffer)

                    # reset digit buffer
                    self._digit_buffer = ''
                    self._disp_mode()
                    did_make_a_change = True

        if did_make_a_change:
            # redraw the content image
            self._update_content()
            self.did_edit = True

    def _on_add_point_click(self, event):
        edt = self._current_edit_mode != 'point'
        self._current_edit_mode = 'point' if edt else None
        self._disp_mode()

    def _on_add_num_click(self, event):
        edt = self._current_edit_mode != 'num'
        self._current_edit_mode = 'num' if edt else None
        self._disp_mode()

    def _on_delete_click(self, event):
        edt = self._current_edit_mode != 'del'
        self._current_edit_mode = 'del' if edt else None
        self._disp_mode()

    def _on_change_color_click(self, event):
        edt = self._current_edit_mode != 'color'
        self._current_edit_mode = 'color' if edt else None
        self._disp_mode()

    def _on_exit_click(self, event):
        if is_correct(self.boxes, self.circles):
            plt.close(self.fig)
        else:
            self.fig.suptitle('Detections don\'t meet specification!', fontsize=16)
            self.fig.canvas.draw_idle()

    def show(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.axis('off')

        self._initialize()
        plt.show()
