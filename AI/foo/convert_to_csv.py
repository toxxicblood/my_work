
import pandas as pd
import os

# Define file paths
directory = "/home/rammah/Documents"
assets_excel_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_excel_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")
assets_csv_path = os.path.join(directory, "assets_with_serials.csv")
villa_csv_path = os.path.join(directory, "villa_register.csv")

try:
    # Convert ASSETS WITH SERIALS.xlsx
    # Read with the correct header row, which we found to be the first row (index 0)
    assets_df = pd.read_excel(assets_excel_path, header=0)
    assets_df.to_csv(assets_csv_path, index=False)
    print(f"Successfully converted '{assets_excel_path}' to CSV.")

    # Convert Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx
    # Read the specific sheet with the correct header row (index 1)
    villa_df = pd.read_excel(villa_excel_path, sheet_name="FA Reg. YEAR2023", header=1)
    villa_df.to_csv(villa_csv_path, index=False)
    print(f"Successfully converted '{villa_excel_path}' to CSV.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the Excel files are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred during conversion: {e}")
