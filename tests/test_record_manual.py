
from pathlib import Path

from scanner.walker import create_file_record


file_path = (
    Path(__file__).parent
    / "data"
    / "sample scan"
    / "photo.jpg"
)

record = create_file_record(file_path)

print("Record:", record)

print("\nIndividual properties:")
print("Name:", record.name)
print("Extension:", record.ext)
print("Category:", record.kind)
print("Size:", record.size)
print("Absolute path:", record.path)

print("\nFuture pipeline fields:")
print("SHA-256:", record.sha256)
print("Perceptual hash:", record.phash)
print("Blur score:", record.blur_score)
