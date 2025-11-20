import cv2
import os
import shutil
import numpy as np

class ImageSaver():
    def __init__(self, 
                 save_dir,
                 temp_dir):
        self.save_dir = save_dir
        self.temp_dir = temp_dir

    def _preprocess_image(self, image_tensor):      
        """Preprocess habitat image tensor for saving."""
        image = image_tensor[:3].cpu().numpy()              # [3, H, W]
        image = image.transpose(1, 2, 0).astype(np.uint8)   # [H, W, 3]
        return image[:, :, ::-1]    # BGR for cv2

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
        suc_episode_dir = os.path.join(self.save_dir, episode_id)
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
                 data_dir):
        pass


