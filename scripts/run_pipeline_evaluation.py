import run_detection_pipeline
from pathlib import Path
import time
import pickle
import datetime

FNAME_EXT_NO_CIRCLES = '-wo-circles'

FONT_SCALES = [2, 3.5, 3, 4, 5, 6, 7]
CIRCLE_DIAMETERS = [1, 2, 3, 6, 10]
iterations = [1, 3, 4, 5]

correct_circles_list = []
correct_text_with_backup_list = []
correct_text_without_backup_list = []
correct_color_list = []
font_scale_list = []
circle_diameter_list = []


def run_iteration(image_path, gt_path):
    with open(gt_path, mode='rb') as input_:
        gt_text_boxes, gt_circles = pickle.load(input_)

    t0 = time.time()
    img_cutted_gs, img_cutted_wb, img_warped_wb = run_detection_pipeline.read_img_and_transform(image_path)

    # don't do circle detection when there are no circles in the image
    do_circle = FNAME_EXT_NO_CIRCLES not in image_path
    text_boxes, circles = run_detection_pipeline.detect_everything(img_cutted_gs, img_cutted_wb, debug=False,
                                                                   do_double_text_detect=False,
                                                                   do_circle_detection=do_circle)
    text_boxes_double, circles_double = run_detection_pipeline.detect_everything(img_cutted_gs, img_cutted_wb,
                                                                                 debug=False,
                                                                                 do_double_text_detect=True,
                                                                                 do_circle_detection=do_circle)

    # test color and circle detection with backup text detection
    correct_color = 0
    correct_tb = 0
    for t in text_boxes_double:
        for other in gt_text_boxes:
            if int(t.text) == int(other.text):
                if abs(t.x - other.x) < 15 and abs(t.y - other.y) < 15:
                    correct_tb += 1
                    if t.color_id == other.color_id:
                        correct_color += 1
    correct_text_with_backup_list.append(correct_tb)
    correct_color_list.append(correct_color)

    # also test text detection without backup detection
    correct_tb = 0
    for t in text_boxes:
        for other in gt_text_boxes:
            if int(t.text) == int(other.text):
                if abs(t.x - other.x) < 15 and abs(t.y - other.y) < 15:
                    correct_tb += 1
    correct_text_without_backup_list.append(correct_tb)

    # test circle detection (it doesn't matter whether this is tested on double or single detection since it runs
    # before the text detection); make sure however that the loaded image has no circles erased
    if FNAME_EXT_NO_CIRCLES in image_path:
        image_path = image_path.replace(FNAME_EXT_NO_CIRCLES, '')
        img_cutted_gs, img_cutted_wb, img_warped_wb = run_detection_pipeline.read_img_and_transform(image_path)
        text_boxes, circles = run_detection_pipeline.detect_everything(img_cutted_gs, img_cutted_wb, debug=False,
                                                                       do_double_text_detect=False)
    correct_circles = 0
    for c in circles:
        for other in gt_circles:
            if abs(c.x - other.x) < 15 and abs(c.y - other.y) < 15:
                correct_circles += 1
    correct_circles_list.append(correct_circles)

    dt = time.time() - t0

    return dt


def main():
    data_dir = Path('../data/test-images').absolute()
    image_paths = [str(data_dir / 'test-hvm - {}.jpg'.format(i)) for i in range(1, 12)]

    print('Running the detection pipeline on {} images.'.format(len(image_paths)))

    for fs in FONT_SCALES:
        for it in iterations:
            print('Running detection with Font scale: {}, iteration: {}.'.format(fs, it))
            # test text detection accuracy independent of circles so load image without circles
            image_path = ('../data/test-images/iat-fs-{}-{}-wo-circles.jpg'.format(fs, it))
            gt_path = Path('../data/test-images/ground-truth/iat-fs-{}-1.pkl'.format(fs))
            print('runtime:', run_iteration(image_path, gt_path), 's')
            font_scale_list.append(fs)
            circle_diameter_list.append(3)

    for cd in CIRCLE_DIAMETERS:
        for it in iterations:
            print('Running detection with circle diameter scale: {}, iteration: {}.'.format(cd, it))
            image_path = ('../data/test-images/iat-cd-{}-{}.jpg'.format(cd, it))
            gt_path = Path('../data/test-images/ground-truth/iat-cd-{}-1.pkl'.format(cd))
            print('runtime:', run_iteration(image_path, gt_path), 's')
            font_scale_list.append(5)
            circle_diameter_list.append(cd)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(Path('~/Desktop/evaluation_{}.csv'.format(timestamp)).expanduser(), mode='w') as out:
        out.write('font_scale, circle_diameter, correct_text_without_backup, '
                  'correct_text_with_backup, correct_circles, correct_colors\n')
        for vals in zip(font_scale_list, circle_diameter_list, correct_text_without_backup_list,
                        correct_text_with_backup_list, correct_circles_list, correct_color_list):
            out.write('{}, {}, {}, {}, {}, {}\n'.format(*vals))


if __name__ == '__main__':
    main()
