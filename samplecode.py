import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

# Load Iris dataset from seaborn's repository
iris = sns.load_dataset('iris')

# Display first 5 rows
print("First 5 rows of the dataset:")
print(iris.head())

# Basic info
print("\nDataset Info:")
print(iris.info())

# Summary statistics
print("\nSummary Statistics:")
print(iris.describe())

# Check for missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Distribution of species
print("\nSpecies Distribution:")
print(iris['species'].value_counts())

# Pairplot for feature relationships
sns.pairplot(iris, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

# Boxplot for each feature by species
for col in iris.columns[:-1]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='species', y=col, data=iris)
    plt.title(f'Boxplot of {col} by Species')
    plt.show()