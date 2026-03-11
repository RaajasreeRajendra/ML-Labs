import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

print("First 5 records:")
print(df.head())

# Identify numerical and categorical columns
print("\nData Types:")
print(df.dtypes)

numerical_cols = df.select_dtypes(include=np.number)
print("\nNumerical Columns:")
print(numerical_cols.columns)

# Experiment 1
# Central Tendency

print("\nMEAN:")
print(numerical_cols.mean())

print("\nMEDIAN:")
print(numerical_cols.median())

print("\nMODE:")
print(numerical_cols.mode())

# Dispersion

print("\nMIN:")
print(numerical_cols.min())

print("\nMAX:")
print(numerical_cols.max())

print("\nSUM:")
print(numerical_cols.sum())

print("\nVARIANCE:")
print(numerical_cols.var())

print("\nSTANDARD DEVIATION:")
print(numerical_cols.std())


# Experiment 2
# Quartiles

print("\nQuartiles:")
print(numerical_cols.quantile([0.25,0.5,0.75]))


# Experiment 3
# Correlation

print("\nCorrelation Matrix:")
print(numerical_cols.corr())

# Covariance

print("\nCovariance Matrix:")
print(numerical_cols.cov())


# Experiment 4
# Histogram

numerical_cols.hist(figsize=(10,6))
plt.suptitle("Histogram of Scores")
plt.show()

# Boxplot

numerical_cols.boxplot()
plt.title("Boxplot of Scores")
plt.show()