import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


REMOVE_ZEROS = True
PRINT_DISTRIBUTIONS = False
SHOW_PLOTS = True

output_tex_filter = '../thesis/contents/x_results_%s.tex'

csv = pd.read_csv("results.csv")
csv.drop(['Start time', 'End time'], axis=1, inplace=True)
csv.dropna(axis=0, how='any', inplace=True)  # Drop rows with NA

processed = csv.groupby(['Dataset', 'Augmentation technique', 'Dataset Reduction (r)']).aggregate(np.mean)
processed.to_csv("results-processed.csv")

dataset_short_names = {"Google Snippets": "snippets",
                       "Tag My News (Titles Only)": "tgn",
                       "TREC": "trec"}

dataset_long_names  = {"Google Snippets": "Google Snippets",
                       "Tag My News (Titles Only)": "TagMyNews",
                       "TREC": "TREC"}

technique_names = {"No Augmentation": "\\textbf{No Augmentation technique}",
                   "Augmentation with max. window size 2": "\\textbf{\emph{Semantic Expansion} with max. window size 2}",
                   "Augmentation with max. window size 3": "\\textbf{\emph{Semantic Expansion} with max. window size 3}"}
table = {}

#if output_tex_filter:
#    # Empty the file.
#    with open(output_tex_filter, 'wt') as f:
#        f.write("")


for dataset in csv['Dataset'].unique():
    dataset_short_name = dataset_short_names[dataset]
    dataset_long_name = dataset_long_names[dataset]

    dataset_rows = csv[csv['Dataset'] == dataset]
    print("Working on dataset %s, row count is %d." % (dataset, len(dataset_rows)))

    alternative_techniques = list(dataset_rows['Augmentation technique'].unique())
    baseline_technique = 'No Augmentation'
    alternative_techniques.remove(baseline_technique)
    all_techniques = [baseline_technique] + alternative_techniques

    # Three boxplots
    f, axes = plt.subplots(len(all_techniques), sharex=True, sharey=True)

    r_values = list(sorted(dataset_rows['Dataset Reduction (r)'].unique()))
    alternate_r_values_strings = ["%.2f" % round(r_value, 2) if i % 2 else "" for i, r_value in enumerate(r_values)]
    alternate_r_values_strings_with_low = ["%.2f" % round(r_value, 2) if i % 2 or i <= 5 else "" for i, r_value in enumerate(r_values)]

    for i, technique in enumerate(all_techniques):
        data = dataset_rows[dataset_rows['Augmentation technique'] == technique]
        accuracy_vectors = [data[data['Dataset Reduction (r)'] == r_value]['Test accuracy'] for r_value in r_values]
        axes[i].boxplot(accuracy_vectors)
        axes[i].set_xlabel("Dataset Reduction (r) value")
        axes[i].set_ylabel("Mean Accuracy (%)")
        axes[i].set_title("%s, %s" % (dataset, technique))
        axes[i].set_xticklabels(alternate_r_values_strings)
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    SHOW_PLOTS and plt.draw()


    # Single multi-line plot

    r_values = list(sorted(dataset_rows['Dataset Reduction (r)'].unique()))

    fig = plt.figure()

    for i, technique in enumerate(all_techniques):
        data = dataset_rows[dataset_rows['Augmentation technique'] == technique]
        accuracy_vectors = [data[data['Dataset Reduction (r)'] == r_value]['Test accuracy'] for r_value in r_values]
        accuracy_means = [np.mean(x) for x in accuracy_vectors]
        p = plt.plot(r_values, accuracy_means)

    plt.legend(all_techniques, loc='upper left')
    plt.xscale("log", nonposx='clip')
    plt.title("%s" % dataset)
    plt.xlabel("Dataset Reduction (r) value")
    plt.ylabel("Mean Test-set Accuracy (%)")
    plt.xticks(r_values, alternate_r_values_strings_with_low)
    plt.setp(p, linewidth=2.0)
    SHOW_PLOTS and plt.draw()



    # t-test
    for alternative_technique in alternative_techniques:
        print("%s: technique=%s, baseline=%s" % (dataset, alternative_technique, baseline_technique))
        table[alternative_technique] = []
        table[baseline_technique] = []

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


            table[alternative_technique].append((r_value, alternative_distribution_mean,
                                                 alternative_distribution_var, p_value))

            table[baseline_technique].append((r_value, baseline_distribution_mean,
                                          baseline_distribution_var, 0))



    if output_tex_filter:


        techniques = list(reversed(list(table.keys())))

        with open(output_tex_filter % ("%s_mean" % dataset_short_name), 'wt') as f:

            table_mean = "table:%s:mean" % dataset_short_name
            table_variance = "table:%s:variance" % dataset_short_name
            table_pvalue = "table:%s:pvalue" % dataset_short_name

            ## MEAN
            output = """   
                Table \\ref{""" + table_mean + """} includes the mean values measured for the 
                dataset, for each of the values of the parameter \\emph{r}, and each of the 
                augmentation methods.
            
                \\begin{small}
                \\begin{tabularx}{\\textwidth}{|X|X|X|X|}
                        \\hline
                        $r$
            """
            for technique in techniques:
                technique = technique_names[technique]
                output += " & %s " % (technique,)

            output += "\\\\\n"
            output += "\hline\n"

            for columns in zip(*(table[i] for i in techniques)):

                r_value = columns[0][0]
                output += " %f " % r_value

                for column in columns:
                    mean = column[1]
                    var = column[2]
                    output += " & $%f$ " % mean
                    #output += " & $Mean = %f$, $\sigma^{2}= %f$ " % (mean, var)

                output += "\\\\\n"

            output += """
                        \hline

                    \end{tabularx}
                                        \end{small}

                    \captionof{table}{Mean values for the """ + dataset_long_name + """ dataset}
                    \label{""" + table_mean + """}
            """
            
            f.write(output)

        with open(output_tex_filter % ("%s_variance" % dataset_short_name), 'wt') as f:
        
            ## VARIANCE
            output = """
               Table \\ref{""" + table_variance + """} includes the variance for the measurements obtained with the 
                dataset, for each of the values of the parameter \\emph{r}, and each of the 
                augmentation methods.
                \\begin{small}

                            \\begin{tabularx}{\\textwidth}{|X|X|X|X|}
    
                        \\hline
    
                        $r$
            """

            for technique in techniques:
                technique = technique_names[technique]
                output += " & %s " % (technique,)

            output += "\\\\\n"
            output += "\hline\n"

            for columns in zip(*(table[i] for i in techniques)):

                r_value = columns[0][0]
                output += " %f " % r_value

                for column in columns:
                    mean = column[1]
                    var = column[2]
                    output += " & $%f$ " % var
                    # output += " & $Mean = %f$, $\sigma^{2}= %f$ " % (mean, var)

                output += "\\\\\n"

            output += """
                        \hline
    
                    \end{tabularx}
                                        \end{small}

                    \captionof{table}{Variance values for the """ + dataset_long_name + """ dataset}
                    \label{""" + table_variance + """}

            """
            
            f.write(output)
            
        with open(output_tex_filter % ("%s_pvalue" % dataset_short_name), 'wt') as f:

            ## SIGNIFICANCE
            output = """
                     Table \\ref{""" + table_variance + """} includes the \emph{p}-values for the Student's \emph{t}-test 
                      performed on the samples obtained when using the dataset, for each of the values of the 
                      parameter \\emph{r}. The values are listed for each of the augmentation methods,
                      when compared to the baseline sample - i.e. with no augmentation.
                                    \\begin{small}

                            \\begin{tabularx}{\\textwidth}{|X|X|X|}
    
                        \\hline
    
                        $r$
            """

            for technique in techniques:
                if technique == "No Augmentation":
                    continue
                technique = technique_names[technique]
                output += " & %s " % (technique,)

            output += "\\\\\n"
            output += "\hline\n"

            for columns in zip(*(table[i] for i in techniques if i != "No Augmentation")):

                r_value = columns[0][0]
                output += " %f " % r_value

                for column in columns:
                    mean = column[1]
                    var = column[2]
                    p_value = column[3]

                    if p_value <= 0.05:
                        output += (" & \\boldmath{$%f$} " % p_value)
                    else:
                        output += (" & $%f$ " % p_value)

                    # output += " & $Mean = %f$, $\sigma^{2}= %f$ " % (mean, var)

                output += "\\\\\n"

            output += """
                        \hline
    
                    \end{tabularx}
                    \end{small}
                    \captionof{table}{\emph{p}-values for the """ + dataset_long_name + """ dataset}
                    \label{""" + table_pvalue + """}

            """


            f.write(output)
        
        print("Tex files updated.")
        
            

plt.show()