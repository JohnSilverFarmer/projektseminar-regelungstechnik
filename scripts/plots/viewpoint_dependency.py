from pathlib import Path
from matplotlib import pyplot as plt

from scripts.plots.utils import read_results, compute_aggregated_stats

EXTENSION = 'pgf'

RESULT_FILE = '../../data/csv-files/eval_results_doppelte_und_einfach_perspektive.csv'

plt.rcParams.update({
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    'text.usetex': True
})


def _aggregate_results(results, with_backup):
    aggregated = {}
    result_key = 'correct_text_with_backup' if with_backup else 'correct_text_without_backup'

    for result in results:
        if result['circle_diameter'] != 3:  # only vary text size
            continue
        viewpoint = result['viewpoint']
        accuracy = result[result_key] / 47.0

        if viewpoint in aggregated:
            aggregated[viewpoint].append(accuracy)
        else:
            aggregated[viewpoint] = [accuracy]

    return aggregated


def main():
    results = read_results(Path(RESULT_FILE))

    ags = _aggregate_results(results, with_backup=False)
    stats = compute_aggregated_stats(ags)

    plt.bar(stats.parameter, stats.metric_mean, yerr=stats.metric_std)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
