from pathlib import Path
from scanner.walker import iter_files

# Test a) empty directory
empty_dir = Path("data/empty_test")
errors_empty = []
files_empty = list(iter_files(empty_dir, errors_empty))
print("=== Test a) Empty directory ===")
print(f"Files found: {len(files_empty)}")
print(f"Errors collected: {len(errors_empty)}")
for e in errors_empty:
    print(f"  - {e}")
print()

# Test b) unicode filenames
unicode_dir = Path("data/unicode_test")
errors_unicode = []
files_unicode = list(iter_files(unicode_dir, errors_unicode))
print("=== Test b) Unicode filenames ===")
print(f"Files found: {len(files_unicode)}")
for f in files_unicode:
    print(f"  - {f.name}")
print(f"Errors collected: {len(errors_unicode)}")
for e in errors_unicode:
    print(f"  - {e}")
print()

# Test c) error collection with custom error list
print("=== Test c) Error collection ===")
# Test with non-existent directory
errors_nonexist = []
try:
    files_nonexist = list(iter_files("data/does_not_exist", errors_nonexist))
    print(f"Files from non-existent dir: {len(files_nonexist)}")
    print(f"Errors collected: {len(errors_nonexist)}")
    for e in errors_nonexist:
        print(f"  - {e}")
except Exception as ex:
    print(f"Exception raised: {type(ex).__name__}: {ex}")
    print(f"Errors in list: {len(errors_nonexist)}")
    for e in errors_nonexist:
        print(f"  - {e}")
print()

# Test with symlink as root (should raise ValueError per implementation)
errors_symlink = []
try:
    # Create a symlink
    import os
    symlink_path = Path("data/symlink_test_root")
    target = Path("data/unicode_test")
    if not symlink_path.exists():
        try:
            os.symlink(target, symlink_path, target_is_directory=True)
        except:
            pass  # Might fail on Windows without admin, skip if fails
    if symlink_path.exists():
        files_symlink = list(iter_files(symlink_path, errors_symlink))
        print(f"Files from symlink root: {len(files_symlink)}")
    print(f"Errors collected: {len(errors_symlink)}")
    for e in errors_symlink:
        print(f"  - {e}")
except Exception as ex:
    print(f"Exception raised: {type(ex).__name__}: {ex}")
    print(f"Errors in list: {len(errors_symlink)}")
    for e in errors_symlink:
        print(f"  - {e}")
print()

# Test with a file as root (should raise NotADirectoryError)
errors_file_root = []
try:
    files_file_root = list(iter_files("data/unicode_test/café.txt", errors_file_root))
    print(f"Files from file as root: {len(files_file_root)}")
except Exception as ex:
    print(f"Exception raised: {type(ex).__name__}: {ex}")
    print(f"Errors in list: {len(errors_file_root)}")
    for e in errors_file_root:
        print(f"  - {e}")
print()
