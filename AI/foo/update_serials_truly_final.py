
import pandas as pd
import os

# Define absolute paths to the files
directory = "/home/rammah/Documents"
assets_serials_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_register_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")

try:
    # Load the Excel files, using the first row as the header
    assets_df = pd.read_excel(assets_serials_path, header=0)
    register_df = pd.read_excel(villa_register_path, sheet_name="FA Reg. YEAR2023", header=1)

    # Strip leading/trailing whitespace from column names
    assets_df.columns = assets_df.columns.str.strip()
    register_df.columns = register_df.columns.str.strip()

    # Ensure the 'SERIAL NUMBER' column exists
    if 'SERIAL NUMBER' not in assets_df.columns:
        assets_df['SERIAL NUMBER'] = ''

    # Initialize a counter for the serial number suffix
    serial_counter = 1
    
    print("Processing items...")

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        area = asset_row["AREA"]
        item = asset_row["ITEM"]
        
        print(f"Searching for: AREA='{area}', ITEM='{item}'")

        # Find the matching item in the register file
        match = register_df[
            (register_df["Location"].str.strip() == str(area).strip()) & 
            (register_df["Item"].str.strip() == str(item).strip())
        ]

        if not match.empty:
            print(f"  -> Match found!")
            # Get the purchase date from the first match
            purchase_date = match.iloc[0]["Date of Purchase"]

            # Check if the purchase_date is a valid datetime object
            if pd.notna(purchase_date) and isinstance(purchase_date, pd.Timestamp):
                year = purchase_date.year
                month = purchase_date.month

                # Format the serial number
                serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"
                print(f"  -> Generated Serial: {serial_number}")

                # Update the 'SERIAL NUMBER' column
                assets_df.at[index, 'SERIAL NUMBER'] = serial_number

                # Increment the counter for the next item
                serial_counter += 1
            else:
                print(f"  -> Match found, but date is invalid: {purchase_date}")
                assets_df.at[index, 'SERIAL NUMBER'] = ''
        else:
            print(f"  -> No match found.")
            # Leave the cell empty if no match is found
            assets_df.at[index, 'SERIAL NUMBER'] = ''

    # Save the updated DataFrame back to the Excel file
    assets_df.to_excel(assets_serials_path, index=False)
    print(f"\nSuccessfully updated '{assets_serials_path}' with the new serial numbers.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both Excel files are in the correct directory.")
except KeyError as e:
    print(f"A column name is incorrect: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
