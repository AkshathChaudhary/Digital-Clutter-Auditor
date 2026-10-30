
from pathlib import Path

from scanner.walker import scan_directory


# Locate the sample directory relative to this script.
folder = (
    Path(__file__).parent
    / "data"
    / "sample scan"
)

# Run the scanner.
records, errors = scan_directory(folder)

# Display results.
print("========== SCAN RESULTS ==========")

print("Total files discovered:", len(records))
print("Total errors:", len(errors))

print("\n========== FILE RECORDS ==========")

for record in records:
    print(
        f"Name: {record.name} | "
        f"Type: {record.kind} | "
        f"Size: {record.size} bytes"
    )

print("\n========== ERRORS ==========")

for error in errors:
    print(error)
