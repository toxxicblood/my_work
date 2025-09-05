
import pandas as pd
import os
from thefuzz import process
from datetime import datetime

# Define absolute paths to the files
directory = "/home/rammah/Documents"
assets_serials_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_register_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")

try:
    # Load the Excel files
    assets_df = pd.read_excel(assets_serials_path, header=0)
    register_df = pd.read_excel(villa_register_path, sheet_name="FA Reg. YEAR2023", header=1)

    # Strip leading/trailing whitespace from column names
    assets_df.columns = assets_df.columns.str.strip()
    register_df.columns = register_df.columns.str.strip()

    # Correctly parse dates with dd/mm/yyyy format
    register_df['Date of Purchase'] = pd.to_datetime(
        register_df['Date of Purchase'], 
        dayfirst=True, 
        errors='coerce'
    )

    # Ensure the 'SERIAL NUMBER' column exists
    if 'SERIAL NUMBER' not in assets_df.columns:
        assets_df['SERIAL NUMBER'] = ''

    # Create a list of choices for fuzzy matching
    register_items = register_df['Item'].astype(str).tolist()
    serial_counter = 1
    
    print("Processing all items and generating serial numbers...")

    # Get current date for fallback
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        item_to_find = asset_row["ITEM"]
        print(f"Processing: '{item_to_find}'")

        best_match, score = process.extractOne(item_to_find, register_items)
        
        year, month = current_year, current_month
        date_source = "current date"

        if score >= 80:
            match_row = register_df[register_df['Item'] == best_match].iloc[0]
            purchase_date = match_row["Date of Purchase"]

            if pd.notna(purchase_date):
                year = purchase_date.year
                month = purchase_date.month
                date_source = f"purchase date ({purchase_date.strftime('%Y-%m-%d')})"

        serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"
        print(f"  -> Generated Serial: {serial_number} (using {date_source})")

        assets_df.at[index, 'SERIAL NUMBER'] = serial_number
        serial_counter += 1

    # Save the updated DataFrame back to the Excel file
    assets_df.to_excel(assets_serials_path, index=False)
    print(f"\nSuccessfully updated '{assets_serials_path}' with serial numbers for all items.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both Excel files are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
