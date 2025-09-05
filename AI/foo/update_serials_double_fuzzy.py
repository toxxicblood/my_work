
import pandas as pd
import os
from thefuzz import process
from datetime import datetime

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

    assets_df['AREA_clean'] = assets_df['AREA'].str.strip().str.lower()
    assets_df['ITEM_clean'] = assets_df['ITEM'].str.strip().str.lower()
    register_df['Location_clean'] = register_df['Location'].astype(str).str.strip().str.lower()
    register_df['Item_clean'] = register_df['Item'].astype(str).str.strip().str.lower()

    register_df['Date of Purchase'] = pd.to_datetime(
        register_df['Date of Purchase'], 
        dayfirst=True, 
        errors='coerce'
    )

    serial_counter = 1
    serial_numbers = []
    
    # Create a list of unique, clean location choices from the register
    location_choices = register_df['Location_clean'].unique()

    print("Processing items with double fuzzy matching (Location -> Item)...")

    current_date = datetime.now()
    current_year, current_month = current_date.year, current_date.month

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        area_to_find = asset_row['AREA_clean']
        item_to_find = asset_row['ITEM_clean']
        
        print(f"Processing: AREA='{asset_row['AREA']}', ITEM='{asset_row['ITEM']}'")

        # --- Step 1: Fuzzy Match Location ---
        best_location_match, loc_score = process.extractOne(area_to_find, location_choices)
        
        year, month = current_year, current_month
        date_source = "current date (no good location match)"

        if loc_score >= 80: # Threshold for a good location match
            # --- Step 2: Filter by the best location match ---
            location_matches = register_df[register_df['Location_clean'] == best_location_match]
            
            # --- Step 3: Fuzzy Match Item within that location ---
            item_choices_in_loc = location_matches['Item_clean'].tolist()
            best_item_match, item_score = process.extractOne(item_to_find, item_choices_in_loc)

            if item_score >= 80: # Threshold for a good item match
                match_row = location_matches[location_matches['Item_clean'] == best_item_match].iloc[0]
                purchase_date = match_row['Date of Purchase']

                if pd.notna(purchase_date):
                    year = purchase_date.year
                    month = purchase_date.month
                    date_source = f"purchase date (found in '{best_location_match}') - {purchase_date.strftime('%Y-%m-%d')}"
                else:
                    date_source = f"current date (match in '{best_location_match}', but date invalid)"
            else:
                date_source = f"current date (location '{best_location_match}' matched, but no good item match)"
        
        serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"
        serial_numbers.append(serial_number)
        print(f"  -> Generated Serial: {serial_number} (Reason: {date_source})")
        serial_counter += 1

    # Update the original Excel file
    final_assets_df = pd.read_excel(assets_excel_path, header=0)
    final_assets_df['SERIAL NUMBER'] = serial_numbers
    final_assets_df.to_excel(assets_excel_path, index=False)

    print(f"\nSuccessfully updated '{assets_excel_path}' with new serial numbers.")

except FileNotFoundError as e:
    print(f"Error: {e}. An Excel file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
