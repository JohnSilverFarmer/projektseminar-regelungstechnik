from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

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

    ags = _aggregate_results(results, with_backup=True)
    stats = compute_aggregated_stats(ags)

    # layout ids are missing number 2 so make 3->2, 4->3 and so on
    stats.parameter[stats.parameter > 1] -= 1
    stats.parameter[:] = stats.parameter.astype(np.int)

    fig, ax = plt.subplots(figsize=(4.135, 3))
    fig.set_size_inches(4.135, 3)

    ax.bar(stats.parameter, stats.metric_mean, yerr=stats.metric_std, error_kw=dict(capsize=6))

    ax.set_xlabel('Perspektive')
    ax.set_ylabel('Genauigkeit')

    plt.tight_layout()
    plt.savefig(Path('~/Desktop/text-erkennung-perspektive.png').expanduser(), bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    main()
