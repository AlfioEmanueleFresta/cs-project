import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


REMOVE_ZEROS = True
PRINT_DISTRIBUTIONS = False
SHOW_PLOTS = True


csv = pd.read_csv("results.csv")
csv.drop(['Start time', 'End time'], axis=1, inplace=True)

processed = csv.groupby(['Dataset', 'Augmentation technique', 'Dataset Reduction (r)']).aggregate(np.mean)
processed.to_csv("results-processed.csv")


for dataset in csv['Dataset'].unique():
    dataset_rows = csv[csv['Dataset'] == dataset]
    print("Working on dataset %s, row count is %d." % (dataset, len(dataset_rows)))

    alternative_techniques = list(dataset_rows['Augmentation technique'].unique())
    baseline_technique = 'No Augmentation'
    alternative_techniques.remove(baseline_technique)
    all_techniques = [baseline_technique] + alternative_techniques

    # Three boxplots
    f, axes = plt.subplots(len(all_techniques), sharex=True, sharey=True)

    r_values = list(sorted(dataset_rows['Dataset Reduction (r)'].unique()))

    for i, technique in enumerate(all_techniques):
        data = dataset_rows[dataset_rows['Augmentation technique'] == technique]
        accuracy_vectors = [data[data['Dataset Reduction (r)'] == r_value]['Test accuracy'] for r_value in r_values]
        axes[i].boxplot(accuracy_vectors)
        axes[i].set_title("%s, %s" % (dataset, technique))

    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    SHOW_PLOTS and plt.show()


    # Single multi-line plot

    r_values = list(sorted(dataset_rows['Dataset Reduction (r)'].unique()))

    for i, technique in enumerate(all_techniques):
        data = dataset_rows[dataset_rows['Augmentation technique'] == technique]
        accuracy_vectors = [data[data['Dataset Reduction (r)'] == r_value]['Test accuracy'] for r_value in r_values]
        accuracy_means = [np.mean(x) for x in accuracy_vectors]
        plt.plot(r_values, accuracy_means)

    plt.legend(all_techniques, loc='upper left')
    plt.xscale("log", nonposx='clip')
    SHOW_PLOTS and plt.show()




    # t-test
    for alternative_technique in alternative_techniques:
        print("%s: technique=%s, baseline=%s" % (dataset, alternative_technique, baseline_technique))
        for r_value in r_values:

            baseline_distribution = dataset_rows[dataset_rows['Augmentation technique'] == baseline_technique]
            baseline_distribution = baseline_distribution[baseline_distribution['Dataset Reduction (r)'] == r_value]
            baseline_distribution = baseline_distribution['Test accuracy'].values
            baseline_distribution = np.array([x for x in baseline_distribution if x > 0]) if REMOVE_ZEROS else baseline_distribution
            baseline_distribution_mean = np.mean(baseline_distribution)
            baseline_distribution_var = np.var(baseline_distribution)

            alternative_distribution = dataset_rows[dataset_rows['Augmentation technique'] == alternative_technique]
            alternative_distribution = alternative_distribution[alternative_distribution['Dataset Reduction (r)'] == r_value]
            alternative_distribution = alternative_distribution['Test accuracy'].values
            alternative_distribution = np.array([x for x in alternative_distribution if x > 0]) if REMOVE_ZEROS else alternative_distribution
            alternative_distribution_mean = np.mean(alternative_distribution)
            alternative_distribution_var = np.var(alternative_distribution)

            appears_better = alternative_distribution_mean > baseline_distribution_mean
            t, p_value = stats.ttest_ind(alternative_distribution, baseline_distribution, axis=0)
            is_significant = appears_better and p_value < 0.05

            print("  r=%f: baseline_mean=%f (var=%f), alternative_mean=%f (var=%f)" % (r_value,
                                                                                       baseline_distribution_mean,
                                                                                       baseline_distribution_var,
                                                                                       alternative_distribution_mean,
                                                                                       alternative_distribution_var),
                  end=", ")
            print("p_value=%f, appears_better=%s, is_significant=%s" % (p_value,
                                                                            appears_better, is_significant))
            PRINT_DISTRIBUTIONS and print("  --> Alternative distribution ", alternative_distribution)
            PRINT_DISTRIBUTIONS and print("  --> Baseline distribution ", baseline_distribution)

