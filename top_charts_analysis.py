import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(data):
    # Drop the rows that have null values
    data = data.dropna()
    
    # Drop the rows that have duplicated values
    data = data.drop_duplicates()
    return data


def analyze_data():
    # Hosted locally at ./data/record_charts.csv
    data = pd.read_csv("./data/record_charts.csv")
    # data = clean_data(data)
    print(data.head())
    
    # Print the number of unique values for each column
    print(data.nunique(), "\n")
    
    print("Info:")
    print(data.info(), "\n")
    
    print("Describe:")
    print(data.describe(), "\n")
    
    print("Columns:")
    print(data.columns, "\n")
    
    print("Shape:")
    print(data.shape, "\n")
    
    
    print(data.dtypes, "\n")

    print("Null values:")    
    print(data.isnull().sum(), "\n")
    print(data[data['url'].isnull()], "\n")
    
    # Drop the rows that have null values in the url column
    data = data.dropna(subset=['url'])
    print("Null values after dropping:")
    print(data[data['url'].isnull()], "\n")

    
    print("Duplicated values:")
    print(data.duplicated().sum(), "\n")
    return data


if __name__ == "__main__":
    analyze_data()