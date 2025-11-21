def get_camera_config(args):
    return {
        "height": args.camera_height, 
        "resolution": [args.env_frame_width, args.env_frame_height], 
        "fov": args.hfov
    }


def get_habitat_camera_config(
        config_path = "envs/habitat/configs/tasks/objectnav_hm3d.yaml"
    ):
    from habitat.config.default import get_config
    config = get_config(config_path)

    camera_height = config.SIMULATOR.AGENT_0.HEIGHT
    camera_resolution = config.SIMULATOR.RGB_SENSOR.WIDTH, config.SIMULATOR.RGB_SENSOR.HEIGHT
    camera_fov = config.SIMULATOR.RGB_SENSOR.HFOV

    print("Camera height:", camera_height)
    print("Camera resolution:", camera_resolution)
    print("Camera fov:", camera_fov)

    return {
        "height": camera_height, 
        "resolution": camera_resolution, 
        "fov": camera_fov
    }