import os

"""
Tool to rename files in a directory with a specified base name and numbering.
The script assumes that the files are named in the same order as they should be numbered.
"""

def rename_files(directory, base_name):
    # Get all file names in the directory
    files = os.listdir(directory)
    # Sort files to maintain any order if necessary (optional)
    files.sort()
    
    # Loop over each file and rename it
    for index, file_name in enumerate(files, start=1):
        # Construct the new file name with padding for the index
        new_name = f"{base_name}_{index:04d}{os.path.splitext(file_name)[1]}"
        # Create full path for both old and new file names
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed '{file_name}' to '{new_name}'")

# Usage
directory_path = 'Classify/RGB_Classify_Dataset4/ripe'
base_name = 'BLP4'
rename_files(directory_path, base_name)
