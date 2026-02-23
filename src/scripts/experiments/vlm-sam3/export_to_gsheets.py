"""
Export experiment summary (same shape as summary_df in experiment.py) to Google Sheets.

Expects SHEETS_KEY in environment: path to Google service account JSON key file.
Share the target sheet with the service account email (Editor).

Usage:
    python export_to_gsheets.py results_train_egoexo_llava.csv

TODO:
* column matching with google sheets file
* finish and test
"""

import os
import argparse

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SHEET_ID = "1Mb5UqYBHxiZ35nbjJF7uTJaIR4euvid3"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def load_dotenv_if_available():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def main():
    load_dotenv_if_available()
    parser = argparse.ArgumentParser(description="Export experiment summary to Google Sheets")
    parser.add_argument("results_csv", help="Path to results_*.csv (full run output)")
    args = parser.parse_args()

    key_path = "../../keysreluminati-research/keys/reluminati-research-ba9d0e9e5c2c.json"

    df = pd.read_csv(args.results_csv)
    summary = df.groupby("experiment")[["iou", "ba", "ca", "le"]].mean().reset_index()
    columns = ["experiment", "iou", "ba", "ca", "le"]

    creds = Credentials.from_service_account_file(key_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1

    rows = [columns] + summary[columns].astype(str).values.tolist()
    sheet.append_rows(rows, value_input_option="USER_ENTERED")
    print(f"Appended {len(summary)} summary rows to sheet {SHEET_ID}")


if __name__ == "__main__":
    main()
