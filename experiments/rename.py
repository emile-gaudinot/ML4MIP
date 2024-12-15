import os
import re

# Define the directory where your files are located
directory = '/data/ML4MIP-Data/preprocessed_data/training/'

# Get all files in the directory
files = os.listdir(directory)

# Iterate through the files
for file in files:
    # Check if the file matches the pattern and does not have '.label'
    if re.match(r'hepaticvessel_.*_patch\[.*\]\.nii\.gz$', file) and not file.endswith('.label.nii.gz'):
        # Create the new file name by appending .img.nii.gz
        new_name = file.replace('.nii.gz', '.img.nii.gz')
        
        # Get the full path for old and new files
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed: {old_path} -> {new_path}')
