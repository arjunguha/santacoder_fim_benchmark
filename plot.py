import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Plot Exact Match Rate by Model, Language, and Mode.')
parser.add_argument('--input', type=str, help='CSV file containing the data')
parser.add_argument('--output', type=str, help='Output filename')
args = parser.parse_args()

# Function to clean the 'Model' column by removing specified prefixes
def clean_model_names(df, prefixes):
    for prefix in prefixes:
        df['Model'] = df['Model'].str.replace(prefix, '')
    return df

# Read the CSV file
data = pd.read_csv(args.input)

# Specify prefixes to remove
prefixes_to_remove = ['sc2-1b-repo-level-ablations-', 'sc2-1b-ablations-']


# Clean the 'Model' names in the dataset
data = clean_model_names(data, prefixes_to_remove)

# Get the unique languages, modes, and models to create subplots
languages = data['Language'].unique()
modes = data['Mode'].unique()
models = data['Model'].unique()

# Create a figure and axes for the subplots
fig, axes = plt.subplots(nrows=len(languages), ncols=1, figsize=(15, 5 * len(languages)))

# Flatten axes array if only one row of subplots
if len(languages) == 1:
    axes = [axes]

# Define the bar width and positions
bar_width = 0.35
index = np.arange(len(models))

# Plot data for each language
for ax, lang in zip(axes, languages):
    # Filter data for the language
    lang_data = data[data['Language'] == lang]

    # Calculate positions for each mode
    bar_positions = [index + bar_width * i for i in range(len(modes))]

    # Plot each mode with different shading (using alpha value)
    for pos, mode in zip(bar_positions, modes):
        mode_data = lang_data[lang_data['Mode'] == mode]
        sorted_data = mode_data.set_index('Model').reindex(models).reset_index()
        ax.bar(pos, sorted_data['Exact Match Rate'], bar_width, label=mode, alpha=0.7)

    # Set plot attributes
    ax.set_title(f'{lang} Language')
    ax.set_ylabel('Exact Match Rate')
    ax.set_xticks(index + bar_width / len(modes))
    ax.set_xticklabels(models, rotation=90)
    ax.legend(title='Mode')

# Adjust layout
plt.tight_layout()
plt.savefig(args.output)