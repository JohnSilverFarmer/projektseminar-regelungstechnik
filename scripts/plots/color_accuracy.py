from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

from scripts.plots.utils import read_results, compute_aggregated_stats

RESULT_FILE = '../../data/csv-files/eval_results_doppelte_text_erkennung.csv'

plt.rcParams.update({
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    'text.usetex': True
})


def _aggregate_results(results):
    aggregated = {}

    for result in results:
        if result['circle_diameter'] != 3:  # only vary text size not circle diameter
            continue

        font_scale = result['font_scale']
        accuracy = result['correct_colors'] / result['correct_texts']

        if font_scale in aggregated:
            aggregated[font_scale].append(accuracy)
        else:
            aggregated[font_scale] = [accuracy]

    return aggregated


def main():
    results = read_results(Path(RESULT_FILE))
    ags = _aggregate_results(results)
    font_scales, accs_means, accs_stds = compute_aggregated_stats(ags)

    fig, ax = plt.subplots(figsize=(4.135, 3))
    fig.set_size_inches(4.135, 3)

    ax.plot(font_scales, accs_means)
    ax.fill_between(font_scales, accs_means - accs_stds, accs_means + accs_stds, alpha=0.4)

    ax.set_xlabel('Buchstabengröße in mm')
    ax.set_ylabel('Genauigkeit')

    ax.grid(True)

    plt.tight_layout()
    plt.savefig(Path('~/Desktop/farb-erkennung-genauigkeit.pgf').expanduser(), bbox_inches='tight')


if __name__ == '__main__':
    main()
