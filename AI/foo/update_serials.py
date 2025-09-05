
import pandas as pd
import os

# Define absolute paths to the files
directory = "/home/rammah/Documents"
assets_serials_path = os.path.join(directory, "ASSETS WITH SERIALS.xlsx")
villa_register_path = os.path.join(directory, "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx")

try:
    # Load the Excel files
    assets_df = pd.read_excel(assets_serials_path, header=1)
    register_df = pd.read_excel(villa_register_path, sheet_name="FA Reg. YEAR2023", header=1)

    # Ensure the 'SERIAL NUMBER' column exists
    if 'SERIAL NUMBER' not in assets_df.columns:
        assets_df['SERIAL NUMBER'] = ''

    # Initialize a counter for the serial number suffix
    serial_counter = 1

    # Iterate through each item in the assets file
    for index, asset_row in assets_df.iterrows():
        area = asset_row["AREA"]
        item = asset_row["ITEM"]

        # Find the matching item in the register file
        match = register_df[
            (register_df["Location"] == area) & (register_df["Item"] == item)
        ]

        if not match.empty:
            # Get the purchase date from the first match
            purchase_date = match.iloc[0]["Date of Purchase"]

            # Check if the purchase_date is a valid datetime object
            if pd.notna(purchase_date) and isinstance(purchase_date, pd.Timestamp):
                year = purchase_date.year
                month = purchase_date.month

                # Format the serial number
                serial_number = f"VPKL/{year}/{month:02d}/{serial_counter:04d}"

                # Update the 'SERIAL NUMBER' column
                assets_df.at[index, 'SERIAL NUMBER'] = serial_number

                # Increment the counter for the next item
                serial_counter += 1
            else:
                # Leave the cell empty if the date is invalid or missing
                assets_df.at[index, 'SERIAL NUMBER'] = ''
        else:
            # Leave the cell empty if no match is found
            assets_df.at[index, 'SERIAL NUMBER'] = ''

    # Save the updated DataFrame back to the Excel file
    assets_df.to_excel(assets_serials_path, index=False)
    print(f"Successfully updated '{assets_serials_path}' with the new serial numbers.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure both Excel files are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

