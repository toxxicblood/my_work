import pandas as pd
import os
from thefuzz import process
from dateutil.parser import parse

# Define absolute paths to the files
directory = "/home/rammah/Documents"
assets_serials_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_register_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")

def parse_date(date_str):
    try:
        return parse(str(date_str))
    except (ValueError, TypeError):
        return None

try:
    # Load the Excel files
    assets_df = pd.read_excel(assets_serials_path, header=0)
    register_df = pd.read_excel(villa_register_path, sheet_name="FA Reg. YEAR2023", header=1)

    # Strip leading/trailing whitespace from column names
    assets_df.columns = assets_df.columns.str.strip()
    register_df.columns = register_df.columns.str.strip()

    # Ensure the 'SERIAL NUMBER' column exists
    if 'SERIAL NUMBER' not in assets_df.columns:
        assets_df['SERIAL NUMBER'] = ''

    # Create a list of choices for fuzzy matching from the register dataframe
    register_items = register_df['Item'].astype(str).tolist()

    # Initialize a counter for the serial number suffix
    serial_counter = 1
    
    print("Processing items with fuzzy matching and robust date parsing...")

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        item_to_find = asset_row["ITEM"]
        
        print(f"Searching for: '{item_to_find}'")

        # Find the best match for the item using fuzzy matching
        best_match, score = process.extractOne(item_to_find, register_items)
        
        if score >= 80:
            print(f"  -> Best match found: '{best_match}' with score {score}")
            
            match_row = register_df[register_df['Item'] == best_match].iloc[0]
            
            purchase_date_str = match_row["Date of Purchase"]
            purchase_date = parse_date(purchase_date_str)

            if purchase_date:
                year = purchase_date.year
                month = purchase_date.month

                serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"
                print(f"  -> Generated Serial: {serial_number}")

                assets_df.at[index, 'SERIAL NUMBER'] = serial_number
                serial_counter += 1
            else:
                print(f"  -> Match found, but date is invalid or missing: {purchase_date_str}")
                assets_df.at[index, 'SERIAL NUMBER'] = ''
        else:
            print(f"  -> No good match found (best score: {score})")
            assets_df.at[index, 'SERIAL NUMBER'] = ''

    # Save the updated DataFrame back to the Excel file
    assets_df.to_excel(assets_serials_path, index=False)
    print(f"\nSuccessfully updated '{assets_serials_path}' with the new serial numbers.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both Excel files are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
