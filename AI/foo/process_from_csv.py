import pandas as pd
import os
from datetime import datetime

# Define file paths
directory = "/home/rammah/Documents"
assets_csv_path = os.path.join(directory, "assets_with_serials.csv")
villa_csv_path = os.path.join(directory, "villa_register.csv")
assets_excel_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx") # For final update

try:
    # Load the CSV files
    assets_df = pd.read_csv(assets_csv_path)
    register_df = pd.read_csv(villa_csv_path)

    # --- Data Cleaning ---
    # Strip whitespace from column names
    assets_df.columns = assets_df.columns.str.strip()
    register_df.columns = register_df.columns.str.strip()

    # Clean the relevant text columns for matching
    assets_df['AREA_clean'] = assets_df['AREA'].str.strip().str.lower()
    assets_df['ITEM_clean'] = assets_df['ITEM'].str.strip().str.lower()
    register_df['Location_clean'] = register_df['Location'].str.strip().str.lower()
    register_df['Item_clean'] = register_df['Item'].str.strip().str.lower()

    # Correctly parse dates with dd/mm/yyyy format
    register_df['Date of Purchase'] = pd.to_datetime(
        register_df['Date of Purchase'], 
        dayfirst=True, 
        errors='coerce'
    )

    serial_counter = 1
    serial_numbers = []
    
    print("Processing items from clean CSV data...")

    current_date = datetime.now()
    current_year, current_month = current_date.year, current_date.month

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        area_clean = asset_row['AREA_clean']
        item_clean = asset_row['ITEM_clean']
        
        print(f"Searching for: AREA='{asset_row['AREA']}', ITEM='{asset_row['ITEM']}'")

        # Find an exact match on the cleaned columns
        match = register_df[
            (register_df['Location_clean'] == area_clean) & 
            (register_df['Item_clean'] == item_clean)
        ]

        year, month = current_year, current_month
        date_source = "current date (no match)"

        if not match.empty:
            purchase_date = match.iloc[0]['Date of Purchase']
            if pd.notna(purchase_date):
                year = purchase_date.year
                month = purchase_date.month
                date_source = f"purchase date ({purchase_date.strftime('%Y-%m-%d')})"
            else:
                date_source = "current date (match found, but date is invalid)"

        serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"
        serial_numbers.append(serial_number)
        print(f"  -> Generated Serial: {serial_number} (using {date_source})")
        serial_counter += 1

    # --- Final Step: Update the original Excel file ---
    # Read the original excel file again to preserve its formatting
    final_assets_df = pd.read_excel(assets_excel_path, header=0)
    # Assign the generated serial numbers
    final_assets_df['SERIAL NUMBER'] = serial_numbers
    # Save the updated data back to the original excel file
    final_assets_df.to_excel(assets_excel_path, index=False)

    print(f"\nSuccessfully updated '{assets_excel_path}' with new serial numbers.")

except FileNotFoundError as e:
    print(f"Error: {e}. A CSV or Excel file was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

finally:
    # Clean up the temporary CSV files
    if os.path.exists(assets_csv_path):
        os.remove(assets_csv_path)
    if os.path.exists(villa_csv_path):
        os.remove(villa_csv_path)
    print("\nCleaned up temporary CSV files.")
