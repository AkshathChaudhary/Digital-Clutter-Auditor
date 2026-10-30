
from pathlib import Path
from scanner.walker import extract_metadata

file_path = Path("data\\sample scan\\Dcouments\\report.pdf")

metadata = extract_metadata(file_path)

for key, value in metadata.items():
    print(f"{key}: {value}")
