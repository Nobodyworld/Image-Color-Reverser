import os
import glob

def rename_files(directory):
    """
    Rename files in the specified directory.
    Files with '_before' and '_after' in their names are paired and renamed with a new pattern.
    
    :param directory: Directory containing the files to be renamed.
    """
    # Convert to an absolute path for reliability
    directory = os.path.abspath(directory)

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Get all files in the directory
    files = glob.glob(f"{directory}/*")

    # Split the files into 'before' and 'after' files
    before_files = [f for f in files if '_before' in f]
    after_files = [f for f in files if '_after' in f]

    # Sort the files
    before_files.sort()
    after_files.sort()

    # Check for matching counts
    if len(before_files) != len(after_files):
        print("The number of 'before' and 'after' files does not match.")
        return

    # Process the files in pairs
    for i in range(len(before_files)):
        for file in (before_files[i], after_files[i]):
            rename_file(file, i+1, directory)

def rename_file(file_path, number, directory):
    """
    Renames a single file according to a new naming pattern.

    :param file_path: The path of the file to be renamed.
    :param number: The sequence number to be used in the new name.
    :param directory: The directory where the file is located.
    """
    basename = os.path.basename(file_path)
    parts = basename.split("_")
    parts[0] = 'animal'
    parts[1] = str(number).zfill(3)  # Pad the number with leading zeros
    new_basename = "_".join(parts)
    new_file = os.path.join(directory, new_basename)
    os.rename(file_path, new_file)
    print(f"Renamed '{basename}' to '{new_basename}'.")

# Use the function
rename_files('.animal')

