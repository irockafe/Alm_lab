import numpy as np
import pandas as pd
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import spearmanr
import seaborn as sns
from multiprocessing import Pool


def select_samples_via_metadata(my_selection, metadata):
    '''
    FUNCTION: Takes a metadata dataframe with columns as categories
                and rows as samples and outputs the sample names
                that meet your defined criteria
    INPUT: my_selection: a dictionary of key-value pairs from metadata
            'metdata: a pandas dataframe containing all metadata
                Ex: {'Tissue': 'Kidney', 'Prognosis': 'Bad'}
    OUTPUT: A list of the samples that meet the selected criteria
                Ex: ['patient 1', patient 12', 'patient 3'] all
                had bad kidneys
    '''

    selected_metadata = metadata  # Not sure if this assignment is necessary
    for key, val in my_selection.iteritems():
        selected_metadata = selected_metadata[(selected_metadata[key] == val)]

    selected_samples = selected_metadata.index
    return list(selected_samples)


def get_phylo_from_RDP(rdp_path, otu, cutoff):
    '''
    FUNTION - Grabs the most specific phylogenetic info that matches your cutoff
    INPUT - rdp_path: the path to your rdp file,
        otu: a string
        cutoff: The probability cutoff, between 0 and 1
    OUTPUT - a string stating the phylogenetic assignment
    '''
    # import rdp, with denovo otus as rows
    column_names = ['empty_1', 'empty_2', 'empty_3', 'empty_4',
                    'Kingdom', 'Kingdom_2', 'Kingdom_score',
                    'Phylum', 'Phylum_2', 'Phylum_score',
                    'Class', 'Class_2', 'Class_score',
                    'Order', 'Order_2', 'Order_score',
                    'Family', 'Family_2', 'Family_score',
                    'Genus', 'Genus_2', 'Genus_score']
    rdp = pd.read_table(rdp_path, sep='\t', index_col=0, names=column_names)

    # Find the most detailed phylogeny that is better than your cutoffs
    phylo_assignment = ''
    phylo_scoring = ['Kingdom_score', 'Phylum_score', 'Class_score',
                     'Order_score', 'Family_score', 'Genus_score']
    for phylo_score in phylo_scoring:
        rdp_score = rdp.loc[otu].loc[phylo_score]
        if rdp_score >= cutoff:
            # Yay, you found a thing!
            phylo_name = phylo_score.replace('_score', '')
            phylo_assignment = '{phylo}: {name}'.format(
                phylo=phylo_name, name=rdp.loc[otu].loc[phylo_name])
            # print 'yaaaaay', rdp_score, phylo_assignment
    return phylo_assignment


def logistic_regression(otu_table, zero_cutoff):
    pass


def log_transform(otu_table_1, zero_add):

    '''
    FUNCTION - converts a counts otu_table to a log-transformed
                fractional otu table
    INPUT - an otu table (species x samples)
    OUTPUT - a log-transformed fractional otu table.
    '''
    # Make fractional
    otu_table = otu_table_1.copy(deep=True)
    counts_per_sample = otu_table.sum(axis=0)
    normalized_otu_table = otu_table.div(counts_per_sample, axis=1)
    # print normalized_otu_table.sum(axis=0).head()
    # iterate through columns of otu table
    # and add 1e-12
    normalized_otu_table += zero_add
    '''
    for idx, column in normalized_otu_table.iteritems():
        zero_indices = column[column == 0].index
        normalized_otu_table.loc[zero_indices, idx] = zero_add
    '''
    # Now apply log_e transform to everything
    log_otu_table = normalized_otu_table.apply(np.log)
    return log_otu_table


def return_abundant_otus(otu_table, abundance_cutoff):
    '''
    INPUT: otu_table: a pandas data frame of counts by samples,
           abundance_cutoff: the fraction an otu must reach to be returned
    FUNCTION: Gets rid of otus that are below some abundance level you
              care about.
    OUTPUT: A pandas dataframe containing all otus above that abundance.
    '''
    sums = otu_table.sum(axis=0)
    frac_otu_table = otu_table.div(sums)
    cutoff_otu_table = frac_otu_table[frac_otu_table > abundance_cutoff]
    # If all data in row is less than cutoff, drop the row
    keep_indices = cutoff_otu_table.dropna(axis=0, how='all').index
    return otu_table.loc[keep_indices]


def spearman_r_rows(otu_table, y, series_name):
    '''
    INPUT - otu_table: a dataframe of fractional otus
            y: pandas series of y-values for spearman (i.e. weightgain).
            series_name: label your p-values
    FUNCTION - Gets a spearman rho and p-value (beware the p-value if
               you have <50 samples for x-val vs. abundance (y).
               x-values are the rows of your otu table.
    OUTPUT - a Series of p-values for each otu
    '''
    # Times ten just so it's obvious that they're not real p-vals
    p_vals = pd.Series(np.ones(otu_table.shape[0])*10, index=otu_table.index,
                       name=series_name)
    rho_vals = pd.Series(np.ones(otu_table.shape[0])*10, index=otu_table.index,
                         name=series_name)

    # Iterate through otus run spearman r
    for i, (name, abundances) in enumerate(otu_table.iterrows()):
        # make sure the right things are being compared
        assert(abundances.index == y.index).all(), \
            "Your sample names don't match up! Are both of them \
        Pandas Series Objects?"
        # ignores the nan values and uses everything else
        rho, pval = spearmanr(abundances, y, nan_policy='omit')
        p_vals.loc[name] = pval
        rho_vals.loc[name] = rho
    return p_vals, rho_vals


def bootstrap_spearman(otu_table, num_bootstraps, y_vals, series_name,
                       plot=False):
    '''
    FUNCTION: Randomly resamples the metadata values (not the abundances),
                for every sample of an otu table.
                Then it calculates the spearman correlation
                for each bootstrap and returns the mean and standard deviation
                of that null distribution for every otu.
    INPUT: otu_table, a pandas df of otu abundances.
        num_bootstraps: The number of resamplings
        y_vals: Your metadata (i.e. weight gain)
        series_name: What you want the output series to be named
        plot: Plots qq-plots of your null distribution and kills the
            script so you can figure out how many bootstraps to run
    OUTPUT:
        a Pandas series, whose index is OTU_name, and values are
        (mean, stdev) of the bootstrapped rho values
    '''
    ones = np.ones(otu_table.shape[0])
    bootstrap_dists = pd.Series({'Mean': ones, 'Std': ones},
                                index=otu_table.index, name=series_name,
                                dtype='object')
    # Get permutations of metadata
    bootstrap_indices = np.zeros(shape=(num_bootstraps, otu_table.shape[1]))
    # Randomly permute the data
    for i in range(0, num_bootstraps):
        bootstrap_indices[i] = np.random.permutation(y_vals)
    # Run spearman on the permuted metadata values for each otu
    for idx, x in otu_table.iterrows():
        rhos = np.zeros(num_bootstraps)
        for i, meta_permutations in enumerate(bootstrap_indices):
            rho, pval = spearmanr(x, meta_permutations)
            rhos[i] = rho
        mean = np.mean(rhos)
        std = np.std(rhos)
        if plot:
            print 'mean: {u}'.format(u=mean)
            print 'std: {std}'.format(std=std)
            qqplot(rhos, line='s')
            sns.plt.show()
            count, bins, ignored = sns.plt.hist(rhos, 30, normed=True)
            y_normal = (1/(std * np.sqrt(2 * np.pi))) * \
                np.exp(-(bins-mean)**2 / (2 * std**2))
            print len(y_normal)
            print len(bins)
            sns.plt.plot(bins, y_normal, color='red', linewidth=2)
            sns.plt.xlabel('Rho values')
            sns.plt.ylabel('Count')
            sns.plt.show()
            raise ValueError("I raised an error because you set plot=True. "
                             "This way you can estimate the number of"
                             "bootstraps needed to get something normal without"
                             "plotting thousands of things ")
        bootstrap_dists.loc[idx] = (mean, std)
    return bootstrap_dists


def bootstrap_spearman_return_rhos(otu_table, num_bootstraps, y_vals,
                                   series_name, plot=False):

    '''
    FUNCTION: Randomly resamples the metadata values (not the abundances),
                for every sample of an otu table.
                Then it calculates the spearman correlation
                for each bootstrap and returns a dataframe containing
                null rho values, the mean and standard deviation of null rhos
                for every otu.
    INPUT: otu_table, a pandas df of otu abundances.
        num_bootstraps: The number of resamplings
        y_vals: Your metadata (i.e. weight gain)
        series_name: What you want the output series to be named
        plot: Plots qq-plots of your null distribution and kills the
            script so you can figure out how many bootstraps to run
    OUTPUT:
        a Pandas dataframe, no index, but fields for: otu_name, experiment_name,
        null_rhos, null_rho_mean, null_rho_std
    '''
    # pre-allocating memory
    # Dataframe that has same number of rows as there are OTUs
    output_df = pd.DataFrame(columns=['OTU', 'Experiment', 'null_rhos',
                                      'null_rho_mean', 'null_rho_std'],
                             index=range(0, otu_table.shape[0]), dtype=object)
    # Get permutations of metadata
    bootstrap_indices = np.zeros(shape=(num_bootstraps, otu_table.shape[1]))
    # Randomly permute the data
    for i in range(0, num_bootstraps):
        bootstrap_indices[i] = np.random.permutation(y_vals)
    # Avoid copying dataframes repeatedly by first getting all the values in
    # a dictionary
    df_base = {'OTU': [], 'Experiment': [], 'null_rhos': [],
               'null_rho_mean': [], 'null_rho_std': []}
    # Run spearman on the permuted metadata values for each otu
    zero_val = min(otu_table)
    print zero_val
    for row_num, [idx, x] in enumerate(otu_table.iterrows()):
        rhos = np.zeros(num_bootstraps)
        for i, meta_permutations in enumerate(bootstrap_indices):
            rho, pval = spearmanr(x, meta_permutations)
            rhos[i] = rho
        mean = np.mean(rhos)
        std = np.std(rhos)
        # Assign output to dataframe
        df_base['OTU'].append(idx)
        df_base['Experiment'].append(series_name)
        df_base['null_rhos'].append(rhos)
        df_base['null_rho_mean'].append(mean)
        df_base['null_rho_std'].append(std)
        # Make qqplots and histograms to check normality
        if plot:
            print 'mean: {u}'.format(u=mean)
            print 'std: {std}'.format(std=std)
            qqplot(rhos, line='s')
            sns.plt.show()
            count, bins, ignored = sns.plt.hist(rhos, 30, normed=True)
            y_normal = (1/(std * np.sqrt(2 * np.pi))) * \
                np.exp(-(bins-mean)**2 / (2 * std**2))
            print len(y_normal)
            print len(bins)
            sns.plt.plot(bins, y_normal, color='red', linewidth=2)
            sns.plt.xlabel('Rho values')
            sns.plt.ylabel('Count')
            sns.plt.show()
            raise ValueError("I raised an error because you set plot=True. "
                             "This way you can estimate the number of"
                             "bootstraps needed to get something normal without"
                             "plotting thousands of things ")
    # make the output dataframe
    output_df = pd.DataFrame(df_base)
    return output_df


def bootstrap_spearman_parallel(otu_table, num_bootstraps, y_vals, series_name,
                                plot=False):
    # NOTE: This is a bit crap and doesn't work well
    '''
    FUNCTION: Randomly resamples the metadata values (not the abundances),
                for every sample of an otu table.
                Then it calculates the spearman correlation
                for each bootstrap and returns the mean and standard deviation
                of that null distribution for every otu.
    INPUT: otu_table, a pandas df of otu abundances.
        num_bootstraps: The number of resamplings
        y_vals: Your metadata (i.e. weight gain)
        series_name: What you want the output series to be named
        plot: Plots qq-plots of your null distribution and kills the
            script so you can figure out how many bootstraps to run
    OUTPUT:
        a Pandas series, whose index is OTU_name, and values are
        (mean, stdev) of the bootstrapped rho values
    '''
    ones = np.ones(otu_table.shape[0])
    bootstrap_dists = pd.Series({'Mean': ones, 'Std': ones},
                                index=otu_table.index, name=series_name,
                                dtype='object')
    # Get permutations of metadata
    bootstrap_indices = np.zeros(shape=(num_bootstraps, otu_table.shape[1]))
    # Randomly permute the data
    for i in range(0, num_bootstraps):
        bootstrap_indices[i] = np.random.permutation(y_vals)
    # Run spearman on the permuted metadata values for each otu
    for idx, x in otu_table.iterrows():
        pool = Pool()
        xy_pairs = [[x] + [i] for i in bootstrap_indices]
        # print xy_pairs[0]
        # print '-------------', spearmanr_one_input(xy_pairs[0])
        # print len(xy_pairs)
        rhos = np.array(pool.map(spearmanr_one_input, xy_pairs))
        # print rhos
        mean = np.mean(rhos)
        std = np.std(rhos)
        pool.close()
        pool.join()

        if plot:
            print 'mean: {u}'.format(u=mean)
            print 'std: {std}'.format(std=std)
            qqplot(rhos, line='s')
            sns.plt.show()
            count, bins, ignored = sns.plt.hist(rhos, 30, normed=True)
            y_normal = (1/(std * np.sqrt(2 * np.pi))) * \
                np.exp(-(bins-mean)**2 / (2 * std**2))
            print len(y_normal)
            print len(bins)
            sns.plt.plot(bins, y_normal, color='red', linewidth=2)
            sns.plt.xlabel('Rho values')
            sns.plt.ylabel('Count')
            sns.plt.show()
            raise ValueError("I raised an error because you set plot=True. "
                             "This way you can estimate the number of"
                             "bootstraps needed to get something normal without"
                             "plotting thousands of things ")
        bootstrap_dists.loc[idx] = (mean, std)
    return bootstrap_dists


def spearmanr_one_input(xy_pair):
    rho, pval = spearmanr(xy_pair, axis=1)
    return rho


def filter_abundance(otu_counts, abundance_cutoff, number_cutoff):
    '''
    FUNCTION: Filters a count otu table (otu x samples) based on desired
        fractional abundance cutoff, and the number of samples an otu must
        be in to not be removed.
        INPUT: otu_counts: table of otu counts (otu x samples)
            abundance cutoff: relative abundance below which you want to filter
            sample_cutoff: number of samples an OTU must be present in
                to not be filtered

    OUTPUT: An otu table with only otus above a given abundance and number of
        samples
    '''
    # If you're trying to require more samples than the number you have,
    # default to the number you have
    if number_cutoff > otu_counts.shape[1]:
        print 'You tried to require more samples than you have. Defaulting to \
            requiring presence in all samples'
        number_cutoff = otu_counts.shape[1]
    sums = otu_counts.sum(axis=0)
    frac_table = otu_counts.div(sums)
    # count number above abundance
    above_cutoff = frac_table > abundance_cutoff
    number_above_cutoff = above_cutoff.sum(axis=1)
    pass_tests_indices = number_above_cutoff[number_above_cutoff >=
                                             number_cutoff].index
    output = otu_counts.loc[pass_tests_indices]
    return output


def make_null_dist(X, y, fxn, name, shuffles, sample_num_threshold,
                   plot=False):
    '''
    INPUT: X: otu table (otus on rows), y: the metadata to permute,
        fxn: the function you will pass to run your test statistics (
           e.g. spearman, mann-whitney, something user-defined),
        name: The name of your grouping, i.e. "Line A WeightGain"
        shuffles: the number of times to shuffle metadata
        sample_num_threshol: The number of samples that must have an otu
            for a distribution to be attempted. This should vary with the
            number of samples you have. i.e. 8choose2 is much smaller than
            16choose2

    FUNCTION: shuffles the y-values shuffles times.

    OUTPUT: a Pandas dataframe, no index, but fields for:
    otu_name, experiment_name, null_rhos, null_rho_mean, null_rho_std
    '''
    # Set up the output
    df_base = {'OTU': np.full(X.shape[0], np.nan, dtype='object'),
               'Experiment': np.full(X.shape[0], np.nan, dtype='object'),
               'true_test_stat': np.full(X.shape[0], np.nan, dtype='float64'),
               'null_test_stats': np.full(X.shape[0], np.nan, dtype='object'),
               'null_mean': np.full(X.shape[0], np.nan, dtype='float64'),
               'null_std': np.full(X.shape[0], np.nan, dtype='float64')}

    # Permute metadata
    # print type(y)
    # if type(y) == list:
    y_array = [y]*shuffles
    # print '\noriginal array:\n', y_array[0:2]
    np.random.seed(1)
    permuted_list = map(np.random.permutation, y_array)
    permuted_y = [pd.Series(lst, index=y.index) for lst in permuted_list]

    # print '\nPermuted y:\n', permuted_y[0:2]
    zero_val = X.min().min()
    # Iterate through each OTU and run test
    for row_num, [idx, x] in enumerate(X.iterrows()):
        # Assign otu and experiment first. This way, you can tell which values
        # get ignored b/c sample size is too small
        df_base['OTU'][row_num] = idx
        df_base['Experiment'][row_num] = name
        # First make sure this otu is present in the minimum number of
        # samples to bother testing it. (If something is present in
        # only one sample, there's no way you'll ever make a null distribution,
        # b/c there will only be as many test-stat values as there are samples)
        zero_samples = sum((x - zero_val) < 1e-12)
        non_zero_samples = X.shape[1] - zero_samples
        if non_zero_samples <= sample_num_threshold:
            continue
        # print 'WUBBALUBBADUBDUB'
        # Now calculate the null test-statistics from permuted y-values
        test_stats = np.zeros(shuffles)
        for i, y_permutation in enumerate(permuted_y):
            # Run your test statistic
            test_stat = fxn(x, y_permutation)
            test_stats[i] = test_stat
        null_mean = np.mean(test_stats)
        null_std = np.std(test_stats)

        # Assign output to dataframe
        df_base['null_test_stats'][row_num] = test_stats
        df_base['null_mean'][row_num] = null_mean
        df_base['null_std'][row_num] = null_std
        df_base['true_test_stat'][row_num] = fxn(x, y)

        # Make qqplots and histograms to check normality
        if plot:
            print 'mean: %s' % null_mean
            print 'std: %s' % null_std
            qqplot(test_stats, line='s')
            sns.plt.show()
            # plot hist
            count, bins, ignored = sns.plt.hist(test_stats, 30, normed=True)
            y_normal = (1/(null_std * np.sqrt(2 * np.pi))) * \
                np.exp(-(bins-null_mean)**2 / (2 * null_std**2))

            print len(y_normal)
            print len(bins)
            sns.plt.plot(bins, y_normal, color='red', linewidth=2)
            sns.plt.xlabel('Test Statistic Value')
            sns.plt.ylabel('Count')
            sns.plt.show()
            raise ValueError("I raised an error because you set plot=True"
                             " This way, you can estimate then number of "
                             "shuffles needed to get a normal distribution "
                             "Without plotting thousands of things")

    output_df = pd.DataFrame(df_base)
    return output_df
