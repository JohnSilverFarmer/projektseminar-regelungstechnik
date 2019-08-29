import numpy as np


def read_results(path):
    """ Load the results from a specified csv file. """
    with open(path, 'r') as f:
        lines = f.readlines()

    names = lines[0].replace(' ', '').replace('\n', '').split(',')
    results = []

    for line in lines[1:]:
        vals = [float(v) for v in line.split(',')]
        results.append(dict(zip(names, vals)))

    return results


def compute_aggregated_stats(ags):
    """ Compute mean and standard, deviation for input aggregated results.

    :param ags: A dictionary where the key is a parameter and values are lists with resulting metric values.
    :return: Tuple of the form (params, metric means, metric stds). Values are sorted according to params.
    """
    params = []
    metric_means = []
    metric_stds = []

    for font_scale, accs in ags.items():
        params.append(font_scale)
        metric_means.append(np.mean(accs))
        metric_stds.append(np.std(accs))

    sort_indices = np.argsort(params)
    params = np.asarray(params)[sort_indices]
    metric_means = np.asarray(metric_means)[sort_indices]
    metric_stds = np.asarray(metric_stds)[sort_indices]
    return params, metric_means, metric_stds
