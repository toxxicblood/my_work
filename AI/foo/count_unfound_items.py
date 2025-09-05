
import pandas as pd
import os
from thefuzz import process
from datetime import datetime

# Define file paths
directory = "/home/rammah/Documents"
assets_excel_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_excel_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")

try:
    # Load and clean data
    assets_df = pd.read_excel(assets_excel_path, header=0)
    register_df = pd.read_excel(villa_excel_path, sheet_name="FA Reg. YEAR2023", header=1)

    assets_df.columns = assets_df.columns.str.strip()
    register_df.columns = register_df.columns.str.strip()

    assets_df['AREA_clean'] = assets_df['AREA'].str.strip().str.lower()
    assets_df['ITEM_clean'] = assets_df['ITEM'].str.strip().str.lower()
    register_df['Location_clean'] = register_df['Location'].astype(str).str.strip().str.lower()
    register_df['Item_clean'] = register_df['Item'].astype(str).str.strip().str.lower()

    register_df['Date of Purchase'] = pd.to_datetime(
        register_df['Date of Purchase'], dayfirst=True, errors='coerce'
    )

    serial_counter = 1
    not_found_counter = 0
    serial_numbers = []
    
    location_choices = register_df['Location_clean'].unique()
    all_item_choices = register_df['Item_clean'].tolist()

    print("Processing items and counting unfound items...")

    current_date = datetime.now()
    current_year, current_month = current_date.year, current_date.month

    for index, asset_row in assets_df.iterrows():
        area_to_find = asset_row['AREA_clean']
        item_to_find = asset_row['ITEM_clean']
        
        found_match = False
        year, month = current_year, current_month
        
        # Pass 1: Location-First
        best_location_match, loc_score = process.extractOne(area_to_find, location_choices)
        if loc_score >= 80:
            location_matches = register_df[register_df['Location_clean'] == best_location_match]
            item_choices_in_loc = location_matches['Item_clean'].tolist()
            if item_choices_in_loc:
                best_item_match, item_score = process.extractOne(item_to_find, item_choices_in_loc)
                if item_score >= 80:
                    match_row = location_matches[location_matches['Item_clean'] == best_item_match].iloc[0]
                    purchase_date = match_row['Date of Purchase']
                    if pd.notna(purchase_date):
                        year, month = purchase_date.year, purchase_date.month
                    found_match = True

        # Pass 2: Item-Only
        if not found_match:
            best_item_match, item_score = process.extractOne(item_to_find, all_item_choices)
            if item_score >= 85:
                match_row = register_df[register_df['Item_clean'] == best_item_match].iloc[0]
                purchase_date = match_row['Date of Purchase']
                if pd.notna(purchase_date):
                    year, month = purchase_date.year, purchase_date.month
                found_match = True

        # Fallback and Counter
        if not found_match:
            not_found_counter += 1

        serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"
        serial_numbers.append(serial_number)
        serial_counter += 1

    # Update the original Excel file
    final_assets_df = pd.read_excel(assets_excel_path, header=0)
    final_assets_df['SERIAL NUMBER'] = serial_numbers
    final_assets_df.to_excel(assets_excel_path, index=False)

    print(f"\nProcessing complete.")
    print(f"Total items processed: {len(assets_df)}")
    print(f"Number of items with a confident match found: {len(assets_df) - not_found_counter}")
    print(f"Number of items NOT found (serial generated with current date): {not_found_counter}")


except FileNotFoundError as e:
    print(f"Error: {e}. An Excel file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
