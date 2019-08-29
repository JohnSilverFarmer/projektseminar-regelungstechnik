from pathlib import Path
from matplotlib import pyplot as plt

from scripts.plots.utils import read_results, compute_aggregated_stats

RESULT_FILE = '../../data/csv-files/eval_results_doppelte_text_erkennung.csv'

plt.rcParams.update({
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    'text.usetex': True
})


def _aggregate_results(results, fixed_font):
    aggregated = {}

    for result in results:
        if fixed_font is not None and result['font_scale'] != fixed_font:
            continue

        if fixed_font is not None:
            circle_diam = result['circle_diameter']
        else:
            circle_diam = result['circle_diameter'] / result['font_scale']

        accuracy = result['correct_circles'] / 42

        if circle_diam in aggregated:
            aggregated[circle_diam].append(accuracy)
        else:
            aggregated[circle_diam] = [accuracy]

    return aggregated


def main():
    results = read_results(Path(RESULT_FILE))

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(0.7*8.27, 3))

    ags = _aggregate_results(results, fixed_font=5)
    font_scales, accs_means, accs_stds = compute_aggregated_stats(ags)

    axs[0].plot(font_scales, accs_means)
    axs[0].fill_between(font_scales, accs_means - accs_stds, accs_means + accs_stds, alpha=0.4)
    axs[0].set_xlabel('Kreisdurchmesser in mm')
    axs[0].set_ylabel('Genauigkeit')

    ags = _aggregate_results(results, fixed_font=None)
    font_to_diam, accs_means, accs_stds = compute_aggregated_stats(ags)

    axs[1].plot(font_to_diam, accs_means)
    axs[1].fill_between(font_to_diam, accs_means - accs_stds, accs_means + accs_stds, alpha=0.4)
    axs[1].set_xlabel('Kreisdurchmesser / Buchstabengröße')

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(Path('~/Desktop/kreis-erkennung-genauigkeit.pgf').expanduser(), bbox_inches='tight')


if __name__ == '__main__':
    main()
