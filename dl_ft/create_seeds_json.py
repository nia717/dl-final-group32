import json
import os
from pathlib import Path

def create_seeds_json(data_dir):
    """
    Create a seeds.json file for the dataset.
    
    The seeds.json file should contain a list of tuples, where each tuple contains:
    1. The name of the prompt directory
    2. A list of seeds (in this case, we'll use 0 as the seed since each prompt directory has 0.jpg and 1.jpg)
    """
    seeds_data = []
    
    # Get all subdirectories in the data directory
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        
        # Check if it's a directory
        if os.path.isdir(subdir_path):
            # Get all prompt directories in the subdirectory
            for prompt_dir in os.listdir(subdir_path):
                prompt_path = os.path.join(subdir_path, prompt_dir)
                
                # Check if it's a directory and contains the required files
                if os.path.isdir(prompt_path) and os.path.exists(os.path.join(prompt_path, "0.jpg")) and \
                   os.path.exists(os.path.join(prompt_path, "1.jpg")) and \
                   os.path.exists(os.path.join(prompt_path, "prompt.json")):
                    
                    # Add the prompt directory and seed to the seeds data
                    # Format: [directory_name, [seed]]
                    relative_path = os.path.join(subdir, prompt_dir)
                    seeds_data.append([relative_path, [0]])
    
    # Write the seeds data to a JSON file
    with open(os.path.join(data_dir, "seeds.json"), "w") as f:
        json.dump(seeds_data, f)
    
    print(f"Created seeds.json with {len(seeds_data)} entries")

if __name__ == "__main__":
    create_seeds_json("ft_data")
