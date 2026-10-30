from pathlib import Path
from scanner.walker import iter_files

folder = Path("data/sample scan")

for file_path in iter_files(folder):
    print(file_path)
