"""
scripts/download_data.py
Acquires, extracts, and validates the official UCI Online Shoppers Purchasing Intention Dataset.
Dataset source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
"""

import os
import sys
import zipfile
import io
import requests
import pandas as pd

EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
]

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
BACKUP_CSV_URL = "https://raw.githubusercontent.com/mwburke/online-shoppers-intention/master/online_shoppers_intention.csv"


def get_data_paths():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    csv_path = os.path.join(raw_dir, "online_shoppers_intention.csv")
    return base_dir, raw_dir, csv_path


def validate_dataset(df: pd.DataFrame) -> bool:
    """Validates structure, row counts, and column types."""
    print(f"[*] Validating dataset shape: {df.shape}")
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"[!] Validation Error: Missing columns: {missing_cols}")
        return False

    if len(df) < 10000:
        print(f"[!] Validation Error: Expected ~12,330 rows, but got {len(df)}")
        return False

    # Check target column
    unique_target = set(df["Revenue"].dropna().unique())
    # May be bool (True, False) or string ('TRUE', 'FALSE', etc.)
    target_bool_check = any(val in [True, False, 1, 0, "TRUE", "FALSE", "True", "False"] for val in unique_target)
    if not target_bool_check:
        print(f"[!] Validation Error: Unexpected target values in Revenue: {unique_target}")
        return False

    print("[+] Dataset validation passed successfully!")
    print(f"    - Total sessions: {len(df):,}")
    print(f"    - Total columns: {len(df.columns)}")
    print(f"    - Missing values total: {df.isnull().sum().sum()}")
    rev_counts = df["Revenue"].value_counts(dropna=False)
    print("    - Target distribution (Revenue):")
    for val, count in rev_counts.items():
        pct = (count / len(df)) * 100
        print(f"        {val}: {count:,} ({pct:.2f}%)")
    return True


def download_and_prepare(force: bool = False):
    base_dir, raw_dir, csv_path = get_data_paths()
    os.makedirs(raw_dir, exist_ok=True)

    if os.path.exists(csv_path) and not force:
        print(f"[+] Dataset already exists at: {csv_path}")
        df = pd.read_csv(csv_path)
        if validate_dataset(df):
            print("[+] Using existing validated dataset.")
            return True
        else:
            print("[!] Existing dataset failed validation. Re-downloading...")

    print(f"[*] Downloading dataset from UCI Repository: {UCI_ZIP_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    downloaded = False
    try:
        resp = requests.get(UCI_ZIP_URL, headers=headers, timeout=30)
        if resp.status_code == 200:
            print("[*] Unpacking zip archive in memory...")
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                # Look for the csv inside zip
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if csv_files:
                    target_name = csv_files[0]
                    with z.open(target_name) as zf:
                        df = pd.read_csv(zf)
                    df.to_csv(csv_path, index=False)
                    downloaded = True
                    print(f"[+] Extracted '{target_name}' to {csv_path}")
    except Exception as e:
        print(f"[!] Primary download failed: {e}")

    if not downloaded:
        print(f"[*] Trying fallback dataset mirror: {BACKUP_CSV_URL}...")
        try:
            resp = requests.get(BACKUP_CSV_URL, headers=headers, timeout=30)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text))
                df.to_csv(csv_path, index=False)
                downloaded = True
                print(f"[+] Downloaded fallback dataset to {csv_path}")
        except Exception as e:
            print(f"[!] Fallback download failed: {e}")

    if not downloaded or not os.path.exists(csv_path):
        print("\n" + "=" * 60)
        print("[!] ERROR: Automatic dataset download could not be completed.")
        print("    Manual Setup Instructions:")
        print("    1. Download 'online_shoppers_intention.csv' from:")
        print("       https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset")
        print(f"    2. Save the CSV file directly to:\n       {csv_path}")
        print("    3. Re-run: python scripts/download_data.py")
        print("=" * 60 + "\n")
        return False

    # Validate the written file
    df = pd.read_csv(csv_path)
    if validate_dataset(df):
        print(f"\n[SUCCESS] Dataset ready at: {csv_path}")
        return True
    else:
        print("[!] Dataset validation failed after writing.")
        return False


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    success = download_and_prepare(force=force_download)
    sys.exit(0 if success else 1)
