
import pandas as pd
import os

# Define file paths
directory = "/home/rammah/Documents"
assets_excel_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_excel_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")

try:
    # Load the Excel files
    assets_df = pd.read_excel(assets_excel_path, header=0)
    register_df = pd.read_excel(villa_excel_path, sheet_name="FA Reg. YEAR2023", header=1)

    # --- Data Cleaning ---
    assets_df.columns = assets_df.columns.str.strip()
    register_df.columns = register_df.columns.str.strip()

    # Get unique, non-empty location names
    asset_locations = assets_df['AREA'].dropna().unique()
    register_locations = register_df['Location'].dropna().unique()

    print("--- Unique Locations in 'ASSETS WITH SERIALS.xlsx' ---")
    for loc in sorted(asset_locations):
        print(loc)

    print("\n--- Unique Locations in 'Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx' ---")
    for loc in sorted(register_locations):
        print(loc)

except FileNotFoundError as e:
    print(f"Error: {e}. An Excel file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

