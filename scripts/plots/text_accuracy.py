from pathlib import Path
from matplotlib import pyplot as plt

from scripts.plots.utils import read_results, compute_aggregated_stats

EXTENSION = 'pgf'

RESULT_FILE = '../../data/csv-files/eval_results_doppelte_und_einfach.csv'

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
        font_scale = result['font_scale']
        accuracy = result[result_key] / 47.0

        if font_scale in aggregated:
            aggregated[font_scale].append(accuracy)
        else:
            aggregated[font_scale] = [accuracy]

    return aggregated


def _apply_basic_style(ax):
    ax.set_xlabel('Buchstabengröße in mm')
    ax.set_ylabel('Genauigkeit')
    ax.grid(True)


def main():
    results = read_results(Path(RESULT_FILE))

    # ===================================
    # First plot without backup detection
    # ===================================
    ags = _aggregate_results(results, with_backup=False)
    font_scales, accs_means, accs_stds = compute_aggregated_stats(ags)

    fig, ax = plt.subplots(figsize=(4.135, 3))
    ax.plot(font_scales, accs_means)
    ax.fill_between(font_scales, accs_means-accs_stds, accs_means+accs_stds, alpha=0.4)

    _apply_basic_style(ax)

    plt.tight_layout()
    plt.savefig(Path('~/Desktop/text-erkennung-genauigkeit-einfach.{}'.format(EXTENSION)).expanduser(),
                bbox_inches='tight')

    # ===============================
    # Plot including backup detection
    # ===============================
    ags = _aggregate_results(results, with_backup=True)
    bup_stats = compute_aggregated_stats(ags)

    fig, ax = plt.subplots(figsize=(4.135, 3))
    colors = ['#1f77b4', '#ff7f0e']

    ax.plot(font_scales, accs_means, color=colors[0], label='Einfach')
    ax.fill_between(font_scales, accs_means - accs_stds, accs_means + accs_stds, alpha=0.4, facecolor=colors[0])

    ax.plot(bup_stats.parameter, bup_stats.metric_mean, color=colors[1], label='Zwei Schritte')
    ax.fill_between(bup_stats.parameter, bup_stats.metric_mean - bup_stats.metric_std,
                    bup_stats.metric_mean + bup_stats.metric_std, alpha=0.4, facecolor=colors[1])

    ax.legend(loc='lower left')

    _apply_basic_style(ax)

    plt.tight_layout()
    plt.savefig(Path('~/Desktop/text-erkennung-genauigkeit-doppelt.{}'.format(EXTENSION)).expanduser(),
                bbox_inches='tight')


if __name__ == '__main__':
    main()
