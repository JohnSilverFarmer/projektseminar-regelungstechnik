import run_detection_pipeline
from pathlib import Path
import time
import pickle

FONT_SCALES = range(2, 8)
CIRCLE_DIAMETERS = [1, 2, 3, 6, 10]
iterations = [1, 3, 4, 5]

correct_circles_list = []
correct_text_list = []
correct_color_list = []
font_scale_list = []
circle_diameter_list = []



def run_iteration(image_path, gt_text_boxes, gt_circles):
    t0 = time.time()
    img_cutted_gs, img_cutted_wb, img_warped_wb = run_detection_pipeline.read_img_and_transform(image_path)
    text_boxes, circles = run_detection_pipeline.detect_everything(img_cutted_gs, img_cutted_wb, False)
    correct_circles = 0
    for c in circles:
        for other in gt_circles:
            x_dev, y_dev = abs(c.x - other.x), abs(c.y - other.y)
            if abs(c.x - other.x) < 20 and abs(c.y - other.y) < 20:
                correct_circles += 1
    correct_circles_list.append(correct_circles)

    correct_color = 0
    correct_tb = 0
    for t in text_boxes:
        for other in gt_text_boxes:
            if int(t.text) == int(other.text):
                if abs(t.x - other.x) < 30 and abs(t.y - other.y) < 30:
                    correct_tb += 1
                    if t.color_id == other.color_id:
                        correct_color += 1
    correct_text_list.append(correct_tb)
    correct_color_list.append(correct_color)

    dt = time.time() - t0

    return dt


def main():
    data_dir = Path('../data/test-images').absolute()
    image_paths = [str(data_dir / 'test-hvm - {}.jpg'.format(i)) for i in range(1, 12)]
    out_files = [str('../data/csv-files/test-hvm - {}.csv'.format(j)) for j in range(1, 12)]

    print('Running the detection pipeline on {} images.'.format(len(image_paths)))

    with open('/Users/eugenrogulenko/Desktop/groundTruth.pkl', mode='r') as input:
        gt_text_boxes, gt_circles = pickle.load(input)

    for fs in FONT_SCALES:
        for it in iterations:
            print('Running detection with Font scale: {}, iteration: {}.'.format(fs, it))
            image_path = ('../data/test-images/iat-fs-{}-{}.jpg'.format(fs, it))
            dt = run_iteration(image_path, gt_text_boxes, gt_circles)
            font_scale_list.append(fs)
            circle_diameter_list.append(3)

    for cd in CIRCLE_DIAMETERS:
        for it in iterations:
            print('Running detection with circle diameter scale: {}, iteration: {}.'.format(cd, it))
            image_path = ('../data/test-images/iat-cd-{}-{}.jpg'.format(cd, it))
            dt = run_iteration(image_path, gt_text_boxes, gt_circles)
            font_scale_list.append(5)
            circle_diameter_list.append(cd)



    with open('/Users/eugenrogulenko/Desktop/eval_results.csv', mode='w') as out:
        out.write('font_scale, circle_diameter, correct_texts, correct_circles, correct_colors\n')
        for fs, cd, txt, circle, color in zip(font_scale_list, circle_diameter_list, correct_text_list, correct_circles_list, correct_color_list):
            out.write('{}, {}, {}, {}, {}\n'.format(fs, cd, txt, circle, color))

    # print('Running the detection pipeline took {:.2f}s per run. Debug was on'.format(dt/len(image_paths)))


if __name__ == '__main__':
    main()
