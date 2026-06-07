import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "qc_reports")
file_path_qc = os.path.join(REPORT_DIR, "qc_report.csv")
df = pd.read_csv(file_path_qc)

# Add expected status based on file path
# Assuming files in "corrupted_data" should be "FAIL" and others should be "PASS"

def expected_status(file_path):
    if "corrupted_data" in file_path:
        return "FAIL"
    else:
        return "PASS"

df["expected_status"] = df["file"].apply(expected_status)
df["is_match"] = df["status"] == df["expected_status"]

total_files = len(df)
matched = df["is_match"].sum()
mismatched = total_files - matched

print("========== QC REPORT SUMMARY ==========")
print(f"Total files      : {total_files}")
print(f"Matched          : {matched}")
print(f"Mismatched       : {mismatched}")

print("\n========== STATUS COUNT ==========")
print(df.groupby(["expected_status", "status"]).size())

if mismatched > 0:
    print("\n========== MISMATCH FILES ==========")
    print(df[df["is_match"] == False][["file", "expected_status", "status", "reasons"]])
else:
    print("\nAll files validated as expected.")

