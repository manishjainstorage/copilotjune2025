import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from the internet
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
iris_df = pd.read_csv(url)

# Display the first few rows of the dataset
print("First 5 rows of the Iris dataset:")
print(iris_df.head())

# Get basic information about the dataset
print("\nDataset Information:")
print(iris_df.info())

# Get descriptive statistics
print("\nDescriptive Statistics:")
print(iris_df.describe())

# Check for missing values
print("\nMissing values per column:")
print(iris_df.isnull().sum())

# Count the occurrences of each species
print("\nSpecies distribution:")
print(iris_df['species'].value_counts())

# Visualize the data
# Pairplot to visualize relationships between features
sns.pairplot(iris_df, hue='species')
plt.suptitle("Pairplot of Iris Dataset by Species", y=1.02)
plt.show()

# Boxplot for each feature by species
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
sns.boxplot(x='species', y='sepal_length', data=iris_df)
plt.title('Sepal Length by Species')

plt.subplot(1, 4, 2)
sns.boxplot(x='species', y='sepal_width', data=iris_df)
plt.title('Sepal Width by Species')

plt.subplot(1, 4, 3)
sns.boxplot(x='species', y='petal_length', data=iris_df)
plt.title('Petal Length by Species')

plt.subplot(1, 4, 4)
sns.boxplot(x='species', y='petal_width', data=iris_df)
plt.title('Petal Width by Species')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris_df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Iris Dataset')
plt.show()