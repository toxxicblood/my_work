
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

    # Clean text columns for matching
    assets_df['AREA_clean'] = assets_df['AREA'].str.strip().str.lower()
    assets_df['ITEM_clean'] = assets_df['ITEM'].str.strip().str.lower()
    register_df['Location_clean'] = register_df['Location'].astype(str).str.strip().str.lower()
    register_df['Item_clean'] = register_df['Item'].astype(str).str.strip().str.lower()

    # Correctly parse dates
    register_df['Date of Purchase'] = pd.to_datetime(
        register_df['Date of Purchase'], 
        dayfirst=True, 
        errors='coerce'
    )

    serial_counter = 1
    serial_numbers = []
    
    print("Processing items with location-first matching logic...")

    current_date = datetime.now()
    current_year, current_month = current_date.year, current_date.month

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        area_to_find = asset_row['AREA_clean']
        item_to_find = asset_row['ITEM_clean']
        
        print(f"Processing: AREA='{asset_row['AREA']}', ITEM='{asset_row['ITEM']}'")

        # --- Step 1: Filter by Location ---
        # Use 'contains' for flexible matching (e.g., 'reception' in 'reception area')
        location_matches = register_df[register_df['Location_clean'].str.contains(area_to_find, na=False)]

        year, month = current_year, current_month
        date_source = "current date (location not found)"
        
        if not location_matches.empty:
            # --- Step 2: Fuzzy Match Item within the Location Pool ---
            item_choices = location_matches['Item_clean'].tolist()
            best_match, score = process.extractOne(item_to_find, item_choices)

            if score >= 80: # Threshold for a good item match
                # Get the full row of the matched item from the location-filtered dataframe
                match_row = location_matches[location_matches['Item_clean'] == best_match].iloc[0]
                purchase_date = match_row['Date of Purchase']

                if pd.notna(purchase_date):
                    year = purchase_date.year
                    month = purchase_date.month
                    date_source = f"purchase date (match found in location) - {purchase_date.strftime('%Y-%m-%d')}"
                else:
                    date_source = "current date (match in location, but date invalid)"
            else:
                date_source = "current date (location found, but no good item match)"

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
