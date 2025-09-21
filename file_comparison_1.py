import os
import hashlib
import time
from tkinter import Tk
from tkinter.filedialog import askdirectory

def get_file_checksum(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def find_files(dir1, dir2):
    """Find duplicate and non-duplicate files in two directories."""
    files_in_dir1 = {}
    files_in_dir2 = {}
    total_files_checked = 0
    start_time = time.time()

    # Calculate checksums for files in the first directory
    for root, _, files in os.walk(dir1):
        for file in files:
            file_path = os.path.join(root, file)
            checksum = get_file_checksum(file_path)
            files_in_dir1[checksum] = file_path
            total_files_checked += 1
            if time.time() - start_time >= 1:
                print(f"Total files checked: {total_files_checked}")
                start_time = time.time()

    # Calculate checksums for files in the second directory
    for root, _, files in os.walk(dir2):
        for file in files:
            file_path = os.path.join(root, file)
            checksum = get_file_checksum(file_path)
            files_in_dir2[checksum] = file_path
            total_files_checked += 1
            if time.time() - start_time >= 2:
                print(f"Total files checked: {total_files_checked}")
                start_time = time.time()

    # Find duplicates and non-duplicates
    duplicates = []
    non_duplicates_dir1 = []
    non_duplicates_dir2 = []

    for checksum, file_path in files_in_dir1.items():
        if checksum in files_in_dir2:
            duplicates.append((file_path, files_in_dir2[checksum]))
        else:
            non_duplicates_dir1.append(file_path)

    for checksum, file_path in files_in_dir2.items():
        if checksum not in files_in_dir1:
            non_duplicates_dir2.append(file_path)

    return duplicates, non_duplicates_dir1, non_duplicates_dir2

def main():
    # Hide the root window
    Tk().withdraw()

    # Ask the user to select the first directory
    dir1 = askdirectory(title="Select the first directory")
    if not dir1:
        print("No directory selected.")
        return

    # Ask the user to select the second directory
    dir2 = askdirectory(title="Select the second directory")
    if not dir2:
        print("No directory selected.")
        return

    # Find duplicate and non-duplicate files
    duplicates, non_duplicates_dir1, non_duplicates_dir2 = find_files(dir1, dir2)

    # Print the results
    if duplicates:
        print("Duplicate files found:") # Print the duplicate files
        # for file1, file2 in duplicates:
        #     print(f"{file1} == {file2}")
    else:
        print("No duplicate files found.")

    if non_duplicates_dir1:
        print("\nNon-duplicate files in the first directory:")
        for file in non_duplicates_dir1:
            print(file)
    else:
        print("\nNo non-duplicate files in the first directory.")

    if non_duplicates_dir2:
        print("\nNon-duplicate files in the second directory:")
        for file in non_duplicates_dir2:
            print(file)
    else:
        print("\nNo non-duplicate files in the second directory.")

if __name__ == "__main__":
    main()