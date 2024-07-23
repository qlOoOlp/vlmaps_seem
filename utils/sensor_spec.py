import numpy as np

REALSENSE_D435_SPEC = {
    "ori_resolution": [1280,720], # [w,h]
    "cropped_resolution": [640, 480], # [w,h]
    "min_depth": 0.3,
    "max_depth": 3,
    "fov_depth_h": 87,
    "fov_depth_v": 58,
    "fov_color_h": 69,
    "fov_color_v": 42,
    #"rot_ro_cam" : np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    "rot_ro_cam" : np.eye(3)
}

OAKD_PRO_SPEC = {
    "ori_resolution": [1280, 800], # [w,h]
    "cropped_resolution": [1280, 720], # [w,h]
    "min_depth": 0.7,
    "max_depth": 3, # origin val : 12
    "fov_depth_h": 80,
    "fov_depth_v": 55,
    "fov_color_h": 69,
    "fov_color_v": 55,
    "color_resolution": [4056, 3040], # [w,h]
    "rot_ro_cam" : np.array([[0,0,1],[-1,0,0],[0,-1,0]])
}

def get_sensor_spec(cam_name):
    sensors = {
        'realsense':REALSENSE_D435_SPEC,
        'oakd':OAKD_PRO_SPEC
    }
    return sensors[cam_name]