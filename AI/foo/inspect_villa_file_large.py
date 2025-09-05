
import pandas as pd
import os

try:
    df = pd.read_excel(os.path.join("/home/rammah/Documents", "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx"), sheet_name="FA Reg. YEAR2023", header=None)
    print(df.head(10))
except FileNotFoundError:
    print("Error: The file 'Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
