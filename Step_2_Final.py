import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


## Import final DataSet

dataset = pd.read_csv('Datasets/Cumulative_dataset.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['time'])
dataset.drop(columns=['time', 'Unnamed: 0'], inplace=True)
dataset.set_index('Timestamp', inplace=True)

#Quantiles and previous day power already there

## Correlation Matrix

correlation = dataset.corr()['kalby_active_power'].drop('kalby_active_power')

# Create a bar plot
plt.figure(figsize=(12, 8))  # Increase figure size for more space
sns.barplot(x=correlation.index, y=correlation.values, palette='viridis')
plt.title('Correlation of Active Power with other features')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=60, ha='right')  # Rotate labels for better visibility and align them to the right
plt.tight_layout()  # Adjust padding to prevent labels from being cut off
plt.savefig('Figures/correlation_plot.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

print('Stop')

