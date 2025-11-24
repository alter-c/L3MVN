import os
import json
from pathlib import Path
import numpy as np

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def count_collected_data(data_root):
    # data_root: images root directory
    data_path = Path(data_root)
    step_count = 0
    episode_count = 0

    for episode_dir in data_path.iterdir():
        if episode_dir.is_dir():
            jpg_files = list(episode_dir.glob("*.jpg"))
            
            steps = len(jpg_files)
            step_count += steps
            episode_count += 1
            
            # print(f"{episode_dir.name}: {steps} steps")
    
    return step_count, episode_count

def stats_step_distance(data_root):
    data_path = Path(data_root)
    step_distances = []

    def distance(step):
        dx = step["x"]
        dy = step["y"]
        return np.sqrt(dx**2 + dy**2)

    json_files = list(data_path.glob("*.json"))
    for f in json_files:
        data = load_json(f)
        for step in data:
            step_d = distance(step["label"]["trajectory"][0])
            step_distances.append(step_d)
        step_distances.pop(-1) 

    avg_distance = np.mean(step_distances)
    max_distance = np.max(step_distances)
    min_distance = np.min(step_distances)

    return avg_distance, max_distance, min_distance


if __name__ == "__main__":
    datasets_root_path = "/home/yongao/repo/L3MVN/datasets"
    images_root = os.path.join(datasets_root_path, "images")
    json_root = os.path.join(datasets_root_path, "objectnav")

    step_count, episode_count = count_collected_data(images_root)
    print(f"Total episodes: {episode_count}, Total steps: {step_count}")

    # avg_distance, max_distance, min_distance = stats_step_distance(json_root)
    # print(f"Step Distance - Avg: {avg_distance:.4f}, Max: {max_distance:.4f}, Min: {min_distance:.4f}")