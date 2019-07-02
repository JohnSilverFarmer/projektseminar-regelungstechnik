import run_detection_pipeline
from pathlib import Path
import time


def main():
    data_dir = Path('../data/test-images').absolute()
    image_paths = [str(data_dir/'test-color-{}.jpg'.format(i)) for i in range(1, 11)]
    out_file = Path.home()/'Desktop'/'test'/'out.csv'  # provides the dir for debug images

    print('Running the detection pipeline on {} images.'.format(len(image_paths)))

    t0 = time.time()
    for image_path in image_paths:
        run_detection_pipeline.main(image_path, out_file, debug=True)
    dt = time.time() - t0

    print('Running the detection pipeline took {:.2f}s per run. Debug was on'.format(dt/len(image_paths)))


if __name__ == '__main__':
    main()
