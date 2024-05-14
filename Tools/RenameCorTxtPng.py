import os

"""
Tool to rename pairs of .txt and .jpg files in a folder with a specified base name and numbering.
The script assumes that the .txt and .jpg files are named in the same order and have the same number.
"""

def rename_files(folder_path, base_name):
    # Get lists of all .txt and .png files
    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    png_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Ensure that the number of txt files matches the number of png files
    if len(txt_files) != len(png_files):
        print(f"Warning: Number of .txt files ({len(txt_files)}) does not match number of .jpg files ({len(png_files)}).")
        return

    # Rename each pair with the specified base name and number
    for i, (txt_file, png_file) in enumerate(zip(txt_files, png_files), start=1):
        new_txt_name = f"{base_name}_{i:04}.txt"
        new_png_name = f"{base_name}_{i:04}.jpg"

        old_txt_path = os.path.join(folder_path, txt_file)
        old_png_path = os.path.join(folder_path, png_file)
        new_txt_path = os.path.join(folder_path, new_txt_name)
        new_png_path = os.path.join(folder_path, new_png_name)

        os.rename(old_txt_path, new_txt_path)
        os.rename(old_png_path, new_png_path)

        print(f"Renamed {txt_file} to {new_txt_name}")
        print(f"Renamed {png_file} to {new_png_name}")

if __name__ == "__main__":
    # Example usage:
    folder = input("Enter the path of the folder containing the files: ")
    base_name = input("Enter the base name to use for renaming: ")

    rename_files(folder, base_name)
