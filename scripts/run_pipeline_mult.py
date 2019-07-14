import run_detection_pipeline
from pathlib import Path
import time


def main():
    data_dir = Path('../data/test-images').absolute()
    image_paths = [str(data_dir/'test-iat - {}.jpg'.format(i)) for i in range(1, 11)]
    out_files = [str('../data/csv-files/test-iat - {}.csv'.format(j)) for j in range(1, 11)]

    print('Running the detection pipeline on {} images.'.format(len(image_paths)))

    t0 = time.time()
    for id, image_path in enumerate(image_paths):
        print('Running detection number: {}.'.format(id+1))
        run_detection_pipeline.main(image_path, out_files[id], debug=True)
    dt = time.time() - t0

    print('Running the detection pipeline took {:.2f}s per run. Debug was on'.format(dt/len(image_paths)))


if __name__ == '__main__':
    main()
