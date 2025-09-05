
import pandas as pd
import os

try:
    df = pd.read_excel(os.path.join("/home/rammah/Documents", "ASSETS WITH SERIALS.xlsx"), header=1)
    print("Column names:", df.columns.tolist())
    print("First 5 rows of data:")
    print(df.head())
except FileNotFoundError:
    print("Error: The file 'ASSETS WITH SERIALS.xlsx' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
