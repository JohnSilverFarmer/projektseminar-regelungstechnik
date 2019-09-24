# Projektseminar Regelungstechnik

Code for the porejct-seminar "Malen nach Zahlen mit dem X-Y-Plotter (Teil Bildverarbeitung)" at TU Darmstadt (SS-19). 
The code in this repository is organized as follows:
- `nbs`: Jupyter notebooks that mainly contain initial experiments and some parameter tuning widgets (e.g. `circle_detection_acc.ipynb`).
- `data`: Test images, csv files and stored debug output.
- `scripts`: Contains the main program (`run_detection_pipeline.py`) and code that produces evaluation results and plots. For more information see the documentation in the source files.

## Installation

We require a working installation of `OpenCV` (3.x  or 4.x), `Tesseract` (Version 4.0, including pytesseract), `Numpy` and `Matplotlib` (Supplies GUI components). It is recommended that you install python packages into a virtual environment.

## Running the Detection 

To run the detection pipeline execute the script `run_detection_pipeline.py` in your preferred terminal. The script will output a header-less csv file containing the ordered list of detected coordinates and line types.
 You need to specify the following parameters:

- `--image`: The file path to the input camera image.
- `--outFile`: A file path where the output csv file should be written to.
- `--debug` (optional): A flag that when present will lead to the script also writing debug images to the same directory as `--outFile`.