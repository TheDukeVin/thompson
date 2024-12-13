import matplotlib.pyplot as plt
import numpy as np



def plot_grouped_bar_chart(species, coverage, ylim=(0,1)):

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, (measurement, error) in coverage.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.errorbar(x + offset, measurement, yerr=error, fmt="o", color="black")
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Coverage prob.')
    ax.set_title('Arm coverage')
    ax.set_xticks(x + width/2, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(ylim[0], ylim[1])

    return fig, ax

if __name__ == '__main__':
    species = ("0.4/0.5", "0.5/0.5")
    coverage = {
        'Arm 1': (0.95, 0.93),
        'Arm 2': (0.96, 0.92),
    }
    plot_grouped_bar_chart(species, coverage)