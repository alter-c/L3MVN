import cv2
import os
import shutil
import json
import math
import time
import numpy as np
from pathlib import Path
from utils.translate import ObjectNavTranslate


class ImageSaver():
    def __init__(self, 
                 base_dir,
                 save_dir,
                 temp_dir):
        self.part_index = 0
        self.base_dir = base_dir
        self.save_dir = save_dir

        self.part_dir = os.path.join(self.base_dir, f"part{self.part_index}", self.save_dir)
        self.temp_dir = os.path.join(base_dir, temp_dir)
        os.makedirs(self.part_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _preprocess_image(self, image_tensor):      
        """Preprocess habitat image tensor for saving."""
        image = image_tensor[:3].cpu().numpy()              # [3, H, W]
        image = image.transpose(1, 2, 0).astype(np.uint8)   # [H, W, 3]
        return image[:, :, ::-1]    # BGR for cv2

    def count_step(self):
        total = 0
        for ep_dir in Path(self.part_dir).iterdir():
            if ep_dir.is_dir():
                total += len(list(ep_dir.iterdir()))
        return total

    def update_part_dir(self):
        self.part_index += 1
        new_part_dir = os.path.join(self.base_dir, f"part{self.part_index}", self.save_dir)
        os.makedirs(new_part_dir, exist_ok=True)
        self.part_dir = new_part_dir
        print(f"Updated save directory to {self.part_dir}")

    def save_temp_image(self, image, episode_id, step, azimuth=0):
        sub_dir = os.path.join(self.temp_dir, f"{episode_id}")
        os.makedirs(sub_dir, exist_ok=True)

        filename = f"{episode_id}_{azimuth:03}_{step:03}.jpg"
        filepath = os.path.join(sub_dir, filename)
        image = self._preprocess_image(image)
        cv2.imwrite(filepath, image)
        # print(f"Saved temp image to {filepath}")

    def save_episode_images(self, episode_id):
        tmp_episode_dir = os.path.join(self.temp_dir, episode_id)
        suc_episode_dir = os.path.join(self.part_dir, episode_id)
        shutil.move(tmp_episode_dir, suc_episode_dir)
        print(f"Saved episode images to {suc_episode_dir}")

    def clean_episode_images(self, episode_id):
        tmp_episode_dir = os.path.join(self.temp_dir, episode_id)
        if os.path.exists(tmp_episode_dir):
            shutil.rmtree(tmp_episode_dir)
        print(f"Cleaned failed episode images in {tmp_episode_dir}")

    def clean_temp_dir(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        print(f"Cleaned temp directory {self.temp_dir}")


class DataSaver():
    def __init__(self, 
                 base_dir,
                 data_dir,
                 camera_config):
        self.base_dir = base_dir
        self.data_dir = data_dir
        
        self.part_index = 0
        self.part_dir = os.path.join(self.base_dir, f"part{self.part_index}", self.data_dir)
        os.makedirs(self.part_dir, exist_ok=True)

        self.height = camera_config["height"]
        self.resolution = camera_config["resolution"]
        self.fov = camera_config["fov"]

        self.task_type = "object_nav"
        self.scene_type = "indoor"
        self.embodiment_type = "wheeled"

        self.predicate_steps = 8

        self.translator = ObjectNavTranslate()

    def update_part_dir(self):
        self.part_index += 1
        new_part_dir = os.path.join(self.base_dir, f"part{self.part_index}", self.data_dir)
        os.makedirs(new_part_dir, exist_ok=True)
        self.part_dir = new_part_dir

    def _visual_input(self, episode_id, step, azimuth=0):
        image_dir = f"images/{episode_id}"
        return [
            f"{image_dir}/{episode_id}_{azimuth:03}_{s:03}.jpg" \
                for s in range(step + 1)
        ]

    def _text_input(self, goal_text_en):
        goal_text_zh = self.translator.translate(goal_text_en)

        article = "an" if goal_text_en[0].lower() in "aeiou" else "a"
        text_input_en = f"Search for {article} {goal_text_en}."
        text_input_zh = f"搜寻{goal_text_zh}。"

        return {
            "en": text_input_en,
            "zh": text_input_zh
        }
    
    def _pose_trans(self, current_pose, target_pose):
        x, y, theta = current_pose
        x_t, y_t, theta_t = target_pose

        c = np.cos(np.deg2rad(theta))
        s = np.sin(np.deg2rad(theta))
        M = np.array([[c, s],
                      [s, -c]])
        dx = x_t - x
        dy = y_t - y

        # coordinate transform
        x_local, y_local = M @ np.array([dx, dy])

        # theta transform
        theta_local = np.deg2rad(theta_t - theta)
        theta_local = theta_local if theta_local >= 0 else theta_local + 2 * np.pi
        
        return x_local, y_local, theta_local

    def _compute_local_trajectory(self, step, trajectory_data, predicate_steps):
        local_trajectory = []
        if step == len(trajectory_data)-1:
            local_trajectory.append({"x": 0.0, "y": 0.0, "z": None, "theta": 0.0})
        else:
            current_pose = trajectory_data[step]
            for target_pose in trajectory_data[step+1: step+1+predicate_steps]:
                x_local, y_local, theta_local = self._pose_trans(current_pose, target_pose)
                local_trajectory.append({"x": x_local, "y": y_local, "z": None, "theta": theta_local})
        return local_trajectory


    def create_step_data(self, episode_id, step, goal_text, trajectory_data, predicate_steps, extra_data=None):
        sample_id = f"{episode_id}_{step:03}"
        local_trajectory = self._compute_local_trajectory(step, trajectory_data, predicate_steps)

        data = {
            "episode_id": episode_id,
            "step": step,
            "sample_id": sample_id,
            "task_type": self.task_type,   
            "metadata": {
                "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scene_type": self.scene_type,
            },

            "embodiment":{
                "type": self.embodiment_type,
                "params": {
                    "max_speed": 1.0,
                    "camera_height": self.height,
                    "camera": {
                        "front": {
                            "azimuth": 0,
                            "resolution": self.resolution,
                            "fov": self.fov,
                        }
                    },
                }
            },

            "visual_input": {
                "front": self._visual_input(episode_id, step, azimuth=0),
            },   

            "text_input": self._text_input(goal_text),

            "label": {
                "trajectory": local_trajectory, 
                "answer": None
            },
            
            "extra": extra_data if extra_data is not None else {},
        }

        return data
    

    def save_episode_data(self, episode_id, goal_text, trajectory_data, extra_data=None):
        episode_data = []
        for step in range(len(trajectory_data)):
            step_data = self.create_step_data(
                episode_id,
                step,
                goal_text,
                trajectory_data,
                self.predicate_steps,
                extra_data
            )
            episode_data.append(step_data)

        filepath = os.path.join(self.part_dir, f"{episode_id}.json")
        with open(filepath, "w") as f:
            json.dump(episode_data, f, ensure_ascii=False, indent=2)

        print(f"Saved episode data to {episode_id}.json")


