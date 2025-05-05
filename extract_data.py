import os
import zipfile
import shutil
from pathlib import Path

def extract_robot_dataset(zip_path='robot_dataset.zip', output_dir='eval_data'):
    """
    extract eval data from robot dataset zip file
    
    Args:
        zip_path: path to the zip file
        output_dir: output directory
    """
    print(f"Strarting to extract robot dataset...")
    
    # create output directory
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # tmp dir
    temp_dir = Path("temp_extract")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    try:
        # unzip
        print(f"Unzip {zip_path} to {temp_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # expected structure: data/instruct-pix2pix-robot/front
        expected_path = temp_dir / "data" / "instruct-pix2pix-robot"
        
        if not expected_path.exists():
            # try to find the correct path
            print("Expected path not found, searching for front and head folders...")
            dataset_paths = []
            for root, dirs, _ in os.walk(temp_dir):
                if 'front' in dirs and 'head' in dirs:
                    dataset_paths.append(Path(root))
                    break
                
            if not dataset_paths:
                raise FileNotFoundError("Error: not found front and head folders in the zip file.")
                
            expected_path = dataset_paths[0]
            
        print(f"Found dataset root: {expected_path}")
            
        # copy front and head folders to output_dir
        front_src = expected_path / "front"
        head_src = expected_path / "head"
        
        front_dst = output_dir / "front"
        head_dst = output_dir / "head"
        
        # if destination exists, remove it
        if front_dst.exists():
            shutil.rmtree(front_dst)
        if head_dst.exists():
            shutil.rmtree(head_dst)
        
        # copy
        print(f"copy front data to {front_dst}...")
        shutil.copytree(front_src, front_dst)
        
        print(f"copy head data to {head_dst}...")
        shutil.copytree(head_src, head_dst)
        
        print(f"Finished extracting robot dataset.")
        
        # statistics
        front_samples = len(list(front_dst.glob("*")))
        head_samples = len(list(head_dst.glob("*")))
        print(f"Data statistics:")
        print(f"  - front View: {front_samples} samples")
        print(f"  - head View: {head_samples} samples")
    
    finally:
        # clean up temp dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    print(f"Data is ready, saved to {output_dir} directory")


if __name__ == "__main__":
    import sys
    
    # 命令行参数处理
    if len(sys.argv) > 1:
        zip_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # 执行数据提取
    extract_robot_dataset(zip_path, output_dir)

# command line usage:
# python extract_data.py path/to/your/robot_dataset.zip path/to/output_dir
