import os
import shutil
import random

def move_percentage_of_pairs(source_dir, target_dir, percentage=10):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Find all files in the source directory
    files = os.listdir(source_dir)

    # Group files into pairs based on their IDs
    pairs = {}
    for file in files:
        if "patch" in file:  # Assumes files with "patch" are the img or label files
            base_id = file.split("_patch")[0]
            if base_id not in pairs:
                pairs[base_id] = []
            pairs[base_id].append(file)

    # Filter out incomplete pairs
    complete_pairs = [pair for pair in pairs.values() if len(pair) == 2]

    # Shuffle and select 10% of the pairs
    random.shuffle(complete_pairs)
    num_to_move = max(1, int(len(complete_pairs) * (percentage / 100)))
    pairs_to_move = complete_pairs[:num_to_move]

    # Move the selected pairs to the target directory
    for pair in pairs_to_move:
        for file in pair:
            shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

    print(f"Moved {len(pairs_to_move)} pairs ({len(pairs_to_move) * 2} files) to {target_dir}.")

# Example usage
source_directory = "/group/cake/preprocessed_data/resize128"
target_directory = "/group/cake/preprocessed_data/resize128_val"
move_percentage_of_pairs(source_directory, target_directory, percentage=10)
