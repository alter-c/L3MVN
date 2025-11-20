from habitat.config.default import get_config


def get_camera_config(
        config_path = "envs/habitat/configs/tasks/objectnav_hm3d.yaml"
    ):
    config = get_config(config_path)

    camera_height = config.SIMULATOR.AGENT_0.HEIGHT
    camera_resolution = config.SIMULATOR.RGB_SENSOR.WIDTH, config.SIMULATOR.RGB_SENSOR.HEIGHT
    camera_fov = config.SIMULATOR.RGB_SENSOR.HFOV

    print("Camera height:", camera_height)
    print("Camera resolution:", camera_resolution)
    print("Camera fov:", camera_fov)

    return camera_height, camera_resolution, camera_fov