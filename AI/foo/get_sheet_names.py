
import openpyxl
import os

try:
    workbook = openpyxl.load_workbook(os.path.join("/home/rammah/Documents", "Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx"))
    print("Sheet names:", workbook.sheetnames)
except FileNotFoundError:
    print("Error: The file 'Villa Physio ASSETS REGISTER - 2023 - 2024.xlsx' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
