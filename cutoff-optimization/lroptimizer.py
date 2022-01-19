"""LROptimizer class. (Log-rank p-value optimizer)."""

from math import ceil

import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test


class LROptimizer:
    """Class allowing easy optimization of hazard prediction cutoffs based on lowest log-rank p-values.

    Three log-rank tests are run for each cutoff pair:
    - Multivariate on all three groups (overall).
    - Low-risk group vs. Mid-risk group (low-mid).
    - Mid-risk group vs. High-risk group (mid-high).

    When searching for cutoffs beneath a certain p-value using `search` method, user can choose to only consider the overall p-value by setting `by_overall=True`. Otherwise all three p-values must fall beneath the passed `max_p`.
    
    Example usage
    -------------
    ```
    # With a DataFrame "df" containing survival information and hazard predictions.
    lro = LROptimizer(df, '*hazard col*', '*events/censor col*', '*survival times col*')

    # Search for cutoffs where no groups contain less than 10% of samples and *all* log-rank tests have p-values less than 0.1.
    lro.search(min_groupsize=.1, max_p=.1, by_overall=False)

    # Plot KM curves for best overall p-value, best low-mid p-value, and best mid-high p-value.
    lro.plot_best()
    ```
    """

    def __init__(
        self, 
        df, 
        hazard_col, 
        events_col, 
        times_col
    ):
        """Create LROptimizer instance.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame containing survival information and hazard predictions.
        hazard_col : str
            Name of column containing hazard predictions.
        events_col : str
            Name of column containing 1s (event occured) and 0s (censored).
        times_col : str
            Name of column containing times (n days) at which the event occured or patient was censored.
        """
        self.df = df.copy()
        self.hazard_col = hazard_col
        self.events_col = events_col
        self.times_col = times_col

        self.df[self.events_col] = self.df[self.events_col].astype(bool)

        self.fig_width = 22
        self.row_height = 7


    def search(
        self, 
        l_range=(-.9, .5), 
        h_max=.95, 
        step=.05, 
        max_p=None, 
        by_overall=False,
        min_groupsize=None,
        max_groupsize=None
    ):
        """Iterate through pairs of hazard prediction cutoffs, running log-rank tests on the resultant patient groupings.
        
        Parameters
        ----------
        l_range : tuple[float, float]
            Min and max values for the low-mid hazard cutoff.
        h_max : float
            Max value for the mid-high hazard cutoff.
        step : float
            Step size for each cutoff iteration.
        max_p : float
            If `by_overall` is True, this is the maximum *overall p-value* below which cutoffs will be saved. Otherwise, it is the maximum value beneath which all p-values (overall, low-mid, and mid-high) must fall.
        by_overall : bool
            Whether to filter cutoff pairs solely by the resulting *overall p-value*.
        min_groupsize : float
            Minimum proportion of samples which any group, defined by cutoffs, is allowed to comprise.
        max_groupsize : float
            Maximum proportion of samples which any group, defined by cutoffs, is allowed to comprise.
        """

        self._reset_best()

        # Low-mid cutoff loop.
        for l in np.arange(l_range[0], l_range[1], step):

            # Mid-high cutoff loop (current low-mid step to mid-high max).
            for h in np.arange(l+step, h_max+step, step):

                # Run tests with these cutoffs and save if results fit the p-value/groupsize parameters.
                self.get_stats(
                    l, h, 
                    max_p=max_p,
                    min_groupsize=min_groupsize,
                    max_groupsize=max_groupsize,
                    save=True,
                    by_overall=by_overall
                )

        # Print summary of results.
        message = f'{len(self.saved)} cutoff pairs found with:'
        if max_p is not None:
            if by_overall:
                message += f'\n- Overall p-value < {max_p}'
            else:
                message += f'\n- All p-values < {max_p}'
        if min_groupsize is not None:
            message += f'\n- No groups with less than {int(100*min_groupsize)}% of samples.'
        if max_groupsize is not None:
            message += f'\n- No groups with more than {int(100*max_groupsize)}% of samples.'

        print(message)


    def get_stats(
        self, 
        cut1, 
        cut2, 
        max_p=None,
        by_overall=False,
        min_groupsize=None, 
        max_groupsize=None, 
        save=False
    ):
        """Run log-rank tests on a single pair of cutoffs.
        
        Parameters
        ----------
        cut1 : float
            Low-mid hazard cutoff.
        cut2 : float
            Mid-high hazard cutoff.
        max_p : float
            If `by_overall` is True, this is the maximum *overall p-value* below which cutoffs will be saved. Otherwise, it is the maximum value beneath which all p-values (overall, low-mid, and mid-high) must fall.
        by_overall : bool
            Whether to filter cutoff pair solely by the resulting *overall p-value*.
        min_groupsize : float
            Minimum proportion of samples which any group, defined by cutoffs, is allowed to comprise.
        max_groupsize : float
            Maximum proportion of samples which any group, defined by cutoffs, is allowed to comprise.
        save : bool, default False
            Whether to save passed cutoffs if they fit p-value/groupsize parameters. If False, test results will simply be returned.
        """

        # Set default filter parameters for those not passed.
        max_p = max_p if (max_p is not None) else 1
        min_groupsize = min_groupsize if (min_groupsize is not None) else 0
        max_groupsize = max_groupsize if (max_groupsize is not None) else 1

        # Group samples by passed cutoffs.
        self.df['group'] = self.df[self.hazard_col].apply(lambda v: self._grouper(v, cut1, cut2))

        # If cutoffs resulted in less than 3 groups, print message and return (skip).
        if self.df['group'].nunique() < 3:
            print(f'Cutoffs [{cut1}, {cut2}] resulted in less than 3 groups. Skipped.')
            return

        # Get normalized group sizes, and if they don't fit parameters, return (skip).
        norm_counts = self.df['group'].value_counts(normalize=True)
        if (norm_counts.min() < min_groupsize) or (norm_counts.max() > max_groupsize):
            return

        # Round cutoffs.
        cut1 = round(cut1, 5)
        cut2 = round(cut2, 5)

        # Create initial results dict.
        stats = {
            'cutoffs': (cut1, cut2),
            'counts': self.df['group'].value_counts(),
            'p-values': {},
            'statistics': {}
        }

        # Run log-rank test on all three groups.
        stats['p-values']['overall'] = multivariate_logrank_test(
            self.df[self.times_col], 
            self.df['group'], 
            self.df[self.events_col]
        ).p_value

        # Run low-mid and mid-high log-rank tests.
        for pair in [['low', 'mid'], ['mid', 'hi']]:

            name = ''.join(pair)

            pop1 = self.df[self.df['group'] == pair[0]]        
            pop2 = self.df[self.df['group'] == pair[1]] 

            result = logrank_test(
                pop1[self.times_col], 
                pop2[self.times_col], 
                pop1[self.events_col], 
                pop2[self.events_col]
            )

            stats['p-values'][name] = result.p_value

        # Create p-value condition depending on whether `by_overall` parameter was True.
        if by_overall:
            save_cond = stats['p-values']['overall'] < max_p
        else:
            save_cond = max(stats['p-values'].values()) < max_p

        # If the p-value condition is met, and `save` parameter is True, save these cutoffs and update "best" attributes.
        if save_cond:
            if save is not False:
                stats['id'] = len(self.saved) + 1
                
                if stats['p-values']['overall'] < self.best_overall[1]:
                    self.best_overall = (stats['id'], stats['p-values']['overall'])
                if stats['p-values']['lowmid'] < self.best_lowmid[1]:
                    self.best_lowmid = (stats['id'], stats['p-values']['lowmid'])
                if stats['p-values']['midhi'] < self.best_midhi[1]:
                    self.best_midhi = (stats['id'], stats['p-values']['midhi'])
                if stats['counts'].std() < self.lowest_std[1]:
                    self.lowest_std = (stats['id'], stats['counts'].std())

                self.saved.append(stats)

            # Return results dict if `save` parameter is False.
            else:
                return stats


    def get_best(self, by):
        """Get best cutoffs from search, by a certain measure.
        
        Parameters
        ----------
        by : str
            One of 'overall' (p-value), 'lowmid' (p-value), 'midhi' (p-value), or 'variance' (lowest group-size variance).
        """

        allowed = {
            'overall': self.best_overall, 
            'lowmid': self.best_lowmid, 
            'midhi': self.best_midhi, 
            'variance': self.lowest_std
        }
        if by not in allowed:
            raise Exception(f'Value for `by` argument must be one of {allowed}')

        stats_id = allowed[by][0]

        for c in self.saved:
            if c['id'] == stats_id:
                return c


    # ---------------------------- KM plotting methods --------------------------- #

    def compare_plots(self, *stats):
        """Plot KM curves for any number of cutoffs.
        
        *stats : strs | Sequences | dicts
            Strings must be one of ['overall', 'lowmid', 'midhi', 'variance']. Sequences must be pairs of cutoffs. Dicts must be actual results dicts from `LROptimizer.get_stats`.
        """

        n_rows = ceil(len(stats) / 3)

        fig_height = (n_rows*self.row_height) if (n_rows != 1) else (.85*self.row_height)
        fig = plt.figure(figsize=[self.fig_width, fig_height])

        for i, c in enumerate(stats, start=1):
            if type(c) == str:
                c = self.get_best(c)
            elif type(c) in [tuple, list]:
                c = self.get_stats(c[0], c[1])

            ax = fig.add_subplot(n_rows, 3, i)
            self._plot(c, ax=ax)

        plt.subplots_adjust(wspace=.3, hspace=.3)


    def plot_saved(self):
        """Plot KM curves for *all* saved cutoff pairs."""

        n_rows = ceil(len(self.saved) / 3)

        fig_height = (n_rows*self.row_height) if (n_rows != 1) else (.85*self.row_height)
        fig = plt.figure(figsize=[self.fig_width, fig_height])

        for i, c in enumerate(self.saved, start=1):

            ax = fig.add_subplot(n_rows, 3, i)
            self._plot(c, ax=ax)

        plt.subplots_adjust(wspace=.3, hspace=.3)


    def plot_best(self, include_lowest_variance=False):
        """Plot KM curves for the best cutoff pairs (by all measures).
        
        Parameters
        ----------
        include_lowest_variance : bool, default False
            Include "lowest group-size variance" as a "best" measure.
        """

        best_in = [self.best_overall, self.best_lowmid, self.best_midhi]
        if include_lowest_variance:
            best_in.append(self.lowest_std)
        best_ids = set([b[0] for b in best_in])

        n_rows = ceil(len(best_ids) / 3)

        fig_height = (n_rows*self.row_height) if (n_rows != 1) else (.85*self.row_height)
        fig = plt.figure(figsize=[self.fig_width, fig_height])

        spi = 1
        for c in self.saved:

            if c['id'] in best_ids:

                ax = fig.add_subplot(n_rows, 3, spi)
                self._plot(c, ax=ax)

                spi += 1

        plt.subplots_adjust(wspace=.3, hspace=.3)


    # ------------------------------ Hidden methods ------------------------------ #

    def _plot(self, stats, ax):
        """Create KM plot given a results dict and a matplotlib Axes."""

        # Always show whole y-range.
        ax.set_ylim([0, 1.1])

        # Generate title.
        title = str(stats['cutoffs'])
        if stats.get('id'):
            if stats['id'] == self.best_overall[0]:
                title += " - Best overall p-value"
            if stats['id'] == self.best_lowmid[0]:
                title += " - Best low-mid p-value"
            if stats['id'] == self.best_midhi[0]:
                title += " - Best mid-high p-value"
            if stats['id'] == self.lowest_std[0]:
                title += " - Lowest variance between group sizes"

        ax.set_title(title)

        # Print p-values.
        ax.text(3800, .26, 'p-values')
        for i, p in enumerate(stats['p-values'], start=1):
            ax.text(3800, .26-i*.05, f"{p:} {round(stats['p-values'][p], 5)}")

        # Group samples by cutoffs in `stats` results dict.
        self.df['group'] = self.df[self.hazard_col].apply(lambda v: self._grouper(v, *stats['cutoffs']))

        # Draw KM curve for each group.
        for group in self.df['group'].unique():
            sub = self.df[self.df['group'] == group]
            count = sub.shape[0]

            kmf = KaplanMeierFitter(label=group)

            kmf.fit(durations=sub[self.times_col], event_observed=sub[self.events_col])

            kmf.plot(show_censors=True, ci_show=False, ax=ax, label=f'{group}: {count}')



    def _grouper(self, v, cut1, cut2):
        """Group a single value as one of ['low', 'mid', 'hi'] based on passed cutoffs."""

        if v < cut1:
            return 'low'
        elif v < cut2:
            return 'mid'
        else:
            return 'hi'


    def _reset_best(self):
        """Remove saved cutoffs and reset "best" attributes."""

        self.saved = []
        self.best_overall = (0, 1)
        self.best_lowmid = (0, 1)
        self.best_midhi = (0, 1)
        self.lowest_std = (0, 10000)