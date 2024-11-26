import numpy as np
from matplotlib import pyplot as plt


def plot_impact(joined, country1=None, country2=None, to_and_from=False, btc_change=False, ax=None):
    single_bar = not country1 and not country2 or not to_and_from and (not country1 or not country2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if not country1 and not country2:
        title = 'All impacts'
        mask1 = joined == joined  # no filtering
        mask2 = mask1
        label1 = 'All'
        label2 = 'All'
    elif (not country1 or not country2) and to_and_from:
        title = f'Impacts for {country1}'
        mask1 = joined.Actor1CountryCode == country1
        mask2 = joined.Actor2CountryCode == country1
        label1 = 'Initiated'
        label2 = 'Received'
    elif not country1 or not country2:
        country = country1 if country1 else country2
        event_type = 'Initiator' if country1 else 'Receiver'
        title = f'{event_type}: {country}'
        mask1 = joined.Actor1CountryCode == country if country1 else joined.Actor2CountryCode == country
        mask2 = mask1
        label1 = event_type
        label2 = event_type
    else:
        title = f'{country1} and {country2}'
        mask1 = (joined.Actor1CountryCode == country1) & (joined.Actor2CountryCode == country2)
        mask2 = (joined.Actor1CountryCode == country2) & (joined.Actor2CountryCode == country1)
        label1 = f'{country1} to {country2}'
        label2 = f'{country2} to {country1}'

    impact1 = joined[mask1].ImpactBin.value_counts(normalize=True)
    impact2 = joined[mask2].ImpactBin.value_counts(normalize=True)
    impact1 = impact1.reindex(joined.ImpactBin.cat.categories, fill_value=0)
    impact2 = impact2.reindex(joined.ImpactBin.cat.categories, fill_value=0)

    if btc_change:
        btc1 = joined[mask1][['ImpactBin', 'Change%']].groupby('ImpactBin', observed=True)['Change%'].mean()
        btc2 = joined[mask2][['ImpactBin', 'Change%']].groupby('ImpactBin', observed=True)['Change%'].mean()
        btc1 = btc1.reindex(joined.ImpactBin.cat.categories, fill_value=0)
        btc2 = btc2.reindex(joined.ImpactBin.cat.categories, fill_value=0)
        title += ' with BTC change'

        ax2 = ax.twinx()
        ax2.set_ylabel('BTC change %')
        ax2.grid(False)
        # ax2.set_yscale('symlog', linthresh=0.01)
        max_btc = max(abs(btc1.max()), abs(btc2.max()) if btc2 is not None else 0)
        ax2.set_ylim(-max_btc * 1.2, max_btc * 1.2)
    else:
        btc1 = None
        btc2 = None
        ax2 = None

    impact_score1 = joined[mask1].Impact.mean()
    impact_score2 = joined[mask2].Impact.mean()
    impact_score1 = 3 + (impact_score1 / 5e3)
    impact_score2 = 3 + (impact_score2 / 5e3)

    x = np.arange(len(impact1))
    width = 0.35

    colors = {'VeryNeg': 'darkred',
              'Neg': 'red',
              'SlightNeg': 'orange',
              'Neutral': 'yellow',
              'SlightPos': 'lightgreen',
              'Pos': 'lime',
              'VeryPos': 'green'}

    if single_bar:
        ax.bar(x, impact1, width * 2, label=label1, color=[colors[cat] for cat in impact1.index])
        ax.axvline(x=impact_score1, color='red', linestyle='--', label=f'Mean impact: {impact_score1 - 3:.2f}')
    else:
        ax.bar(x - width / 2, impact1, width, label=label1, color=[colors[cat] for cat in impact1.index])
        ax.bar(x + width / 2, impact2, width, label=label2, color=[colors[cat] for cat in impact2.index], alpha=0.7)
        ax.axvline(x=impact_score1, color='red', linestyle='--', label=f'{label1} mean: {impact_score1 - 3:.2f}')
        ax.axvline(x=impact_score2, color='cyan', linestyle='--', label=f'{label2} mean: {impact_score2 - 3:.2f}')

    if btc_change:
        y_0 = 0  # Use 0 as baseline instead of middle of plot
        ax2.axhline(y=y_0, color='gray', linestyle=':', alpha=0.3, label='0% change')

        def plot_btc_indicator(x_pos, change, impact):
            if impact > 0:
                if abs(change) <= 0.05:
                    change = 0
                    # Plot dot for small changes
                    ax2.plot(x_pos, change, 'ko', markersize=4)
                else:
                    # Plot arrow for larger changes
                    ax2.annotate(f'{change:.2f}%',
                                 xy=(x_pos, y_0),
                                 xytext=(x_pos, change),
                                 color='black',
                                 ha='center',
                                 va='bottom' if change > 0 else 'top',
                                 arrowprops=dict(
                                     arrowstyle='<-',
                                     color='lime' if change > 0 else 'red',
                                     connectionstyle='arc3,rad=0',
                                     shrinkA=0,
                                     shrinkB=0
                                 ))

        # Plot BTC changes for impact bins
        for i, (change1, change2) in enumerate(zip(btc1, btc2 if not single_bar else [None] * len(btc1))):
            if not single_bar:
                plot_btc_indicator(i - width / 2, change1, impact1.iloc[i])
                plot_btc_indicator(i + width / 2, change2, impact2.iloc[i])
            else:
                plot_btc_indicator(i, change1, impact1.iloc[i])

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(impact1.index, rotation=60)
    ax.legend(loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel('Proportion')
    ax.set_xlabel('Impact')

    return ax


def plot_impacts(joined, countries=None, country1=None, country2=None, to_and_from=False, btc_change=False):
    """
    Plot impact distribution for countries or between two countries
    :param joined: joined GDELT and BTC data
    :param countries: plot impacts for multiple countries. if country1 and country2 are None, plot for all countries individually
    :param country1: if countries is provided, plot impacts from this country to each, otherwise plot only for this country or between this and country2
    :param country2: if countries is provided, plot impacts from each country to this, otherwise plot only for this country or between country1 and this
    :param to_and_from: if True, plot impacts from country1 to country2 and from country2 to country1
    :param btc_change: if True, plot BTC price change on a secondary y-axis
    """
    if not countries:
        plot_impact(joined, country1, country2, to_and_from, btc_change)
    elif country1 and country2:
        raise ValueError('Cannot provide countries and country1 and country2 at the same time')
    else:
        rows = (len(countries) + 1) // 2  # Calculate needed rows
        fig, axes = plt.subplots(rows, 2, figsize=(15, 5 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, country in enumerate(countries):
            if country1:
                plot_impact(joined, country1, country, to_and_from, btc_change, axes[i])
            elif country2:
                plot_impact(joined, country, country2, to_and_from, btc_change, axes[i])
            else:
                plot_impact(joined, country, to_and_from=to_and_from, btc_change=btc_change, ax=axes[i])

        # Remove empty subplots if odd number of countries
        if len(countries) % 2 == 1:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        return fig
