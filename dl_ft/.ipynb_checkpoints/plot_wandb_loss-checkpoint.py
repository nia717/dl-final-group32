import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# wandb 日志目录
wandb_dir = "logs/train_default/wandb/offline-run-20250504_221714-train_default/files"

print(f"Searching for loss data in: {wandb_dir}")

# 查找 wandb 日志文件
def find_wandb_files():
    # 尝试查找 wandb-history.jsonl 文件
    history_files = glob.glob(os.path.join(wandb_dir, "wandb-history.jsonl"))
    if history_files:
        return history_files[0], "history"
    
    # 尝试查找 wandb-events.jsonl 文件
    events_files = glob.glob(os.path.join(wandb_dir, "wandb-events.jsonl"))
    if events_files:
        return events_files[0], "events"
    
    # 尝试查找 wandb-summary.json 文件
    summary_files = glob.glob(os.path.join(wandb_dir, "wandb-summary.json"))
    if summary_files:
        return summary_files[0], "summary"
    
    return None, None

# 从 wandb-history.jsonl 文件中提取 loss
def extract_from_history(file_path):
    print(f"Extracting loss from history file: {file_path}")
    
    steps = []
    losses = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # 查找 loss 相关的键
                loss_keys = [k for k in data.keys() if 'loss' in k.lower()]
                
                if loss_keys and '_step' in data:
                    steps.append(data['_step'])
                    
                    for key in loss_keys:
                        if key not in losses:
                            losses[key] = []
                        losses[key].append(data[key])
            except json.JSONDecodeError:
                continue
    
    return steps, losses

# 从 wandb-events.jsonl 文件中提取 loss
def extract_from_events(file_path):
    print(f"Extracting loss from events file: {file_path}")
    
    steps = []
    losses = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                if 'historyStep' in data and 'data' in data:
                    step_data = data['data']
                    
                    # 查找 loss 相关的键
                    loss_keys = [k for k in step_data.keys() if 'loss' in k.lower()]
                    
                    if loss_keys:
                        steps.append(data['historyStep'])
                        
                        for key in loss_keys:
                            if key not in losses:
                                losses[key] = []
                            losses[key].append(step_data[key])
            except json.JSONDecodeError:
                continue
    
    return steps, losses

# 从 wandb-summary.json 文件中提取 loss
def extract_from_summary(file_path):
    print(f"Extracting loss from summary file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 查找 loss 相关的键
        loss_keys = [k for k in data.keys() if 'loss' in k.lower()]
        
        if loss_keys:
            print("Found loss keys in summary:")
            for key in loss_keys:
                print(f"  {key}: {data[key]}")
            
            return True
    except json.JSONDecodeError:
        print("Error decoding JSON from summary file")
    
    return False

# 查找并处理 wandb 日志文件
file_path, file_type = find_wandb_files()

if file_path is None:
    print("No wandb log files found.")
    exit(1)

# 根据文件类型提取 loss
if file_type == "history":
    steps, losses = extract_from_history(file_path)
    
    if not losses:
        print("No loss data found in history file.")
        exit(1)
    
    print(f"Found {len(steps)} steps with loss data.")
    print(f"Loss metrics: {list(losses.keys())}")
    
    # 绘制每个 loss 指标的曲线
    for key in losses:
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses[key])
        plt.xlabel('Steps')
        plt.ylabel(key)
        plt.title(f'Training {key}')
        plt.grid(True)
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{key}_curve_{timestamp}.png"
        plt.savefig(filename)
        print(f"Loss curve saved to {filename}")
        
        # 打印统计信息
        print(f"Min {key}: {min(losses[key]):.6f}")
        print(f"Max {key}: {max(losses[key]):.6f}")
        print(f"Last {key}: {losses[key][-1]:.6f}")

elif file_type == "events":
    steps, losses = extract_from_events(file_path)
    
    if not losses:
        print("No loss data found in events file.")
        exit(1)
    
    print(f"Found {len(steps)} steps with loss data.")
    print(f"Loss metrics: {list(losses.keys())}")
    
    # 绘制每个 loss 指标的曲线
    for key in losses:
        plt.figure(figsize=(12, 6))
        plt.plot(steps, losses[key])
        plt.xlabel('Steps')
        plt.ylabel(key)
        plt.title(f'Training {key}')
        plt.grid(True)
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{key}_curve_{timestamp}.png"
        plt.savefig(filename)
        print(f"Loss curve saved to {filename}")
        
        # 打印统计信息
        print(f"Min {key}: {min(losses[key]):.6f}")
        print(f"Max {key}: {max(losses[key]):.6f}")
        print(f"Last {key}: {losses[key][-1]:.6f}")

elif file_type == "summary":
    success = extract_from_summary(file_path)
    
    if not success:
        print("No loss data found in summary file.")
        exit(1)
else:
    print(f"Unknown file type: {file_type}")
    exit(1)