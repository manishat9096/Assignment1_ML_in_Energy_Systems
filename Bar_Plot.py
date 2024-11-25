import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

metrics_df = pd.read_csv('metrics.csv')

### PLOTS RECALL

# Separate the data for each generator
generators = metrics_df['Label'].unique()

# Create a bar plot for each generator
for generator in generators:
    # Filter the dataframe for the current generator
    generator_df = metrics_df[metrics_df['Label'] == generator]

    # Pivot the data for plotting
    pivoted = generator_df.pivot(index='Hour', columns='Model', values='Recall')

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(pivoted.index))  # Hours (0 to 23)
    width = 0.15  # Width of each bar

    # Iterate through models and plot bars
    for i, model in enumerate(pivoted.columns):
        ax.bar(x + i * width, pivoted[model], width, label=model)

    # Add labels, title, and legend
    ax.set_xlabel('Hour', fontsize=14)
    ax.set_ylabel('Recall', fontsize=14)
    ax.set_title(f'Recall per Model by Hour for Generator {generator}', fontsize=16)
    ax.set_xticks(x + 2 * width)  # Center the ticks
    ax.set_xticklabels(pivoted.index, fontsize=12)
    ax.legend(title='Model', fontsize=12)

    # Show the plot
    plt.tight_layout()
    fig.savefig(f"Figures/{generator}_Recall_by_Hour.png")
    plt.close(fig)

print('Precision')

# PLOTS PRECISON

# Create a bar plot for each generator
for generator in generators:
    # Filter the dataframe for the current generator
    generator_df = metrics_df[metrics_df['Label'] == generator]

    # Pivot the data for plotting
    pivoted = generator_df.pivot(index='Hour', columns='Model', values='Precision')

    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(pivoted.index))  # Hours (0 to 23)
    width = 0.15  # Width of each bar

    # Iterate through models and plot bars
    for i, model in enumerate(pivoted.columns):
        ax.bar(x + i * width, pivoted[model], width, label=model)

    # Add labels, title, and legend
    ax.set_xlabel('Hour', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(f'Precision per Model by Hour for Generator {generator}', fontsize=16)
    ax.set_xticks(x + 2 * width)  # Center the ticks
    ax.set_xticklabels(pivoted.index, fontsize=12)
    ax.legend(title='Model', fontsize=12)

    plt.tight_layout()
    fig.savefig(f"Figures/{generator}_Precision_by_Hour.png")
    plt.close(fig)

print('End')