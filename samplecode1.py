import pandas as pd

url = "https://raw.githubusercontent.com/manishjainstorage/Datasets/main/Flipkart.csv"
df = pd.read_csv(url, encoding='latin1')

print(df.head())