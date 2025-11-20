import time
import yaml

def create_episode_data(
        episode_id, 
        step,
        sample_id,
        camera_height,
        fov,
        front_visual_input,
        text_input_en,
        text_input_zh,
        trajectory
    ):

    data = {
            "episode_id": episode_id,
            "step": step,
            "sample_id": sample_id,
            "task_type": "object_nav",

            "metadata": {
                "collection_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "scene_type": "indoor"
            },

            "embodiment":{
                "type": "wheeled",
                "params": {
                    "max_speed": 1.0,
                    "camera_height": camera_height, # Random
                    "camera": {
                        "front": {
                            "azimuth": 0,
                            "resolution": [
                                960,
                                720
                            ],
                            "fov": fov, # Random
                        }
                    },
                }
            },

            "visual_input": {
                "front": front_visual_input,
                # 文件命名规则: %episode_id%_%azimuth:03%_%step:03%
                # [
                    # "xxx/%episode_id%_000_000.jpg", 
                    # "xxx/%episode_id%_000_001.jpg",
                # ]
            }, # 历史图像输入

            "text_input": {
                "en": text_input_en, 
                "zh": text_input_zh
            },

            "label": {
                "trajectory": trajectory,
                # [
                    # {
                    #   "x": 0.0,
                    #   "y": 0.0,
                    #   "z": None,
                    #   "theta": 0.0,
                    # }, ...
                # ]
            }, # 未来待预测点

            "answer": None
        }
    
    return data

def camera_config():
    from habitat.config.default import get_config

    config_path = "envs/habitat/configs/tasks/objectnav_hm3d.yaml"
    config = get_config(config_path)

    camera_height = config.SIMULATOR.AGENT_0.HEIGHT
    camera_resolution = config.SIMULATOR.RGB_SENSOR.WIDTH, config.SIMULATOR.RGB_SENSOR.HEIGHT
    camera_fov = config.SIMULATOR.RGB_SENSOR.HFOV

    print("Camera height:", camera_height)
    print("Camera resolution:", camera_resolution)
    print("Camera fov:", camera_fov)