import os, json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils.lseg_utils import get_lseg_feats
from utils.mapping_utils import save_map, load_pose, load_map, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point, load_depth_npy
from utils.mapping_utils import depth2pc_with_fov2, get_sim_cam_mat_with_fov2
from utils.preprocess import png_to_npy
from utils.get_transform import get_transform

# sensor
from utils.sensor_spec import get_sensor_spec
from utils.mapping_utils import calculate_fov_cropped

# LSeg
from lseg.modules.models.lseg_net import LSegEncNet
# SAM
import open_clip
from utils.extract_conceptfusion_features_vlmaps import extract_conceptfusion
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

def update_maps_sim(save_paths, lists, color_top_down_height):
    # load paths & lists
    color_top_down_save_path, grid_save_path, weight_save_path, obstacles_save_path, param_save_path = save_paths

    color_top_down = load_map(color_top_down_save_path)
    grid = load_map(grid_save_path)
    weight = load_map(weight_save_path)
    obstacles = load_map(obstacles_save_path)

    # read the param json
    rgb_list, depth_list, pose_list = lists

    # read hparams
    with open(param_save_path, 'r') as f:
        args = json.load(f)

    # Step1. Load appropriate models
    if not args['openclip']:
        # LSeg
        model = LSegEncNet(args['lang'], arch_option=0,
                            block_depth=0,
                            activation='lrelu',
                            crop_size=args['crop_size'])
        model_state_dict = model.state_dict()

        lseg_pretrained_state_dict = torch.load(args['lseg_ckpt'])
        lseg_pretrained_state_dict = {k.lstrip('net.'): v for k, v in lseg_pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(lseg_pretrained_state_dict)
        model.load_state_dict(lseg_pretrained_state_dict)

        model.eval()
        model = model.cuda()

        # lseg features
        labels = args['lang'].split(",")
        transform, _MEAN, _STD = get_transform()
    else:
        # Conceptfusion
        # Step1. Load SAM model with Automatic mask generation options
        # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
        print("Loading SAM...")
        sam = sam_model_registry['_'.join(args['sam_ckpt'].split("_")[1:3])](checkpoint=Path(os.path.join(args['sam_root_dir'], args['sam_ckpt'])))
        sam.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            points_per_batch=16,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        # Step2. Load CLIP model
        # https://github.com/mlfoundations/open_clip
        print("Loading OpenCLIP...")
        print(
            f"Initializing OpenCLIP model: {args['clip_version']}"
            f" pre-trained on {args['openclip_pretrained_dataset']}..."
        )
        openclip, _, openclip_preprocess = open_clip.create_model_and_transforms(
            args['clip_version'], args['openclip_pretrained_dataset']
        )
        openclip.cuda()
        openclip.eval()

    # load & transform
    tf_list = []
    data_iter = zip(rgb_list, depth_list, pose_list)
    pbar = tqdm(total=len(rgb_list))

    # load all images and depths and poses per one image
    for data_sample in data_iter:
        rgb_path, depth_path, pose_path = data_sample

        # read rgb
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pos, rot = load_pose(pose_path)
        rot_ro_cam = np.eye(3)              
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        rot = rot @ rot_ro_cam 
                
        pos[1] += args['camera_height']        
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0])

        tf = init_tf_inv @ pose

        # read depth
        depth = load_depth_npy(depth_path)

        # read semantic
        # semantic = load_semantic_npy(semantic_path)
        # semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        ##############################################################
        # Extract VLFM features
        # (1, embedding_dim, H, W)
        if not args['openclip']:
            # (1, embedding_dim, 347(390), 520) - float32
            _, features, _ = get_lseg_feats(model, rgb, labels, args['crop_size'], args['base_size'], transform, _MEAN, _STD)
        else:
            # (1, embedding_dim, 347(390), 520) - float32
            features = extract_conceptfusion(mask_generator, openclip, openclip_preprocess, rgb_path)
        ##############################################################

        # Step2. depth to local point cloud
        pc, mask = depth2pc(depth)

        # sampling
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::args['depth_sample_rate']]
        
        mask = mask[shuffle_mask]   
        pc = pc[:, shuffle_mask]    
        pc = pc[:, mask]            

        # Step2-1. local to global point cloud
        pc_global = transform_pc(pc, tf)

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])               # ((720, 1080, 3)         
        feat_cam_mat = get_sim_cam_mat(features.shape[2], features.shape[3])    # ((1, 512, 347, 520))

        # Step3. project all the global point cloud onto the ground                     
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(args['gs'], args['cs'], p[0], p[2])

            # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            # Step4. rgb embedding vector
            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            # semantic_v = semantic[rgb_py, rgb_px]
            # if semantic_v == 40:
            #     semantic_v = -1

            # when the projected location is already assigned a color value before, overwrite if the current point has larger height
            if p_local[1] < color_top_down_height[y, x]:
                color_top_down[y, x] = rgb_v
                color_top_down_height[y, x] = p_local[1]
                # gt[y, x] = semantic_v

            # average the visual embeddings if multiple points are projected to the same grid cell
            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= features.shape[3] or py >= features.shape[2]):
                feat = features[0, :, py, px]
                # feat = np.array(feat)
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1

            # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
            if p_local[1] > args['camera_height']:
                continue
            obstacles[y, x] = 0
        pbar.update(1)

    save_map(color_top_down_save_path, color_top_down)
    # save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)

def update_maps(save_paths, lists, color_top_down_height):
    # load paths & lists
    color_top_down_save_path, grid_save_path, weight_save_path, obstacles_save_path, param_save_path = save_paths

    color_top_down = load_map(color_top_down_save_path)
    grid = load_map(grid_save_path)
    weight = load_map(weight_save_path)
    obstacles = load_map(obstacles_save_path)

    # read the param json
    rgb_list, depth_list = lists

    # read hparams
    with open(param_save_path, 'r') as f:
        args = json.load(f)

    # Step1. Load appropriate models
    if not args['openclip']:
        # load LSeg model
        model = LSegEncNet(args['lang'], arch_option=0,
                            block_depth=0,
                            activation='lrelu',
                            crop_size=args['crop_size'])
        model_state_dict = model.state_dict()

        lseg_pretrained_state_dict = torch.load(args['lseg_ckpt'])
        lseg_pretrained_state_dict = {k.lstrip('net.'): v for k, v in lseg_pretrained_state_dict['state_dict'].items()}
        model_state_dict.update(lseg_pretrained_state_dict)
        model.load_state_dict(lseg_pretrained_state_dict)

        model.eval()
        model = model.cuda()

        # lseg features
        labels = args['lang'].split(",")
        transform, _MEAN, _STD = get_transform()
    else: 
        # Conceptfusion
        # Step1. Load SAM model with Automatic mask generation options
        # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
        print("Loading SAM...")
        sam = sam_model_registry['_'.join(args['sam_ckpt'].split("_")[1:3])](checkpoint=Path(os.path.join(args['sam_root_dir'], args['sam_ckpt'])))
        sam.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            points_per_batch=16,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        # Step2. Load CLIP model
        # https://github.com/mlfoundations/open_clip
        print("Loading OpenCLIP...")
        print(
            f"Initializing OpenCLIP model: {args['clip_version']}"
            f" pre-trained on {args['openclip_pretrained_dataset']}..."
        )
        openclip, _, openclip_preprocess = open_clip.create_model_and_transforms(
            args['clip_version'], args['openclip_pretrained_dataset']
        )
        openclip.cuda()
        openclip.eval()

    # load a sensor spec
    sensor = get_sensor_spec(args['cam_name'])
    fov_h_cropped = calculate_fov_cropped(sensor['fov_depth_h'], sensor['ori_resolution'][0], sensor['cropped_resolution'][0])
    fov_v_cropped = calculate_fov_cropped(sensor['fov_depth_v'], sensor['ori_resolution'][1], sensor['cropped_resolution'][1])

    # load & transform
    tf_list = []
    pose_dir = os.path.join(args['img_save_dir'], "pose")
    pose_txt = os.path.join(pose_dir, f"{args['pose']}.txt")   # 한 줄씩 읽도록
    f = open(pose_txt, "r")
    f.readline()
    pbar = tqdm(total=len(rgb_list))

    # load all images and depths and poses per one image
    for data_sample in zip(rgb_list, depth_list, f.readlines()):
        rgb_path, depth_path, pose_path = data_sample

        # read rgb
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        pos, rot = load_pose(pose_path, flag=False) # habitat: z backward, y upward, x to the right
        rot_ro_cam = np.eye(3)              
        rot = rot @ rot_ro_cam                      # 기존 ros를 camera coordinate로 바꿔줌
        
        pos[1] += (args['camera_height']+1)         # 기존 pos는 y upward 이므로 여기에 camera_height를 더함
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0])

        tf = init_tf_inv @ pose

        # read depth
        depth = png_to_npy(depth_path)
        # depth = load_depth_npy(depth_path)

        # read semantic
        # semantic = load_semantic_npy(semantic_path)
        # semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        ##############################################################
        # Extract VLFM features
        if not args['openclip']:
            _, features, _ = get_lseg_feats(model, rgb, labels, args['crop_size'], args['base_size'], transform, _MEAN, _STD)
        else:
            features = extract_conceptfusion(mask_generator, openclip, openclip_preprocess, rgb_path)
        ##############################################################
        
        # Step2. depth to local point cloud
        pc, mask = depth2pc_with_fov2(depth, min_depth=sensor['min_depth'], max_depth=sensor['max_depth'], fov_h=fov_h_cropped, fov_v=fov_v_cropped)#,min_depth=0.001#,min_depth=0.01), max_depth=0.2)
        # pc, mask = depth2pc(depth)

        # sampling
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::args['depth_sample_rate']]
        
        mask = mask[shuffle_mask]   
        pc = pc[:, shuffle_mask]    
        pc = pc[:, mask]            

        # Step2-1. local to global point cloud
        pc_global = transform_pc(pc, tf)

        rgb_cam_mat = get_sim_cam_mat_with_fov2(rgb.shape[0], rgb.shape[1], fov_h=fov_h_cropped, fov_v=fov_v_cropped) #1080, 1920,
        feat_cam_mat = get_sim_cam_mat_with_fov2(features.shape[2], features.shape[3], fov_h=fov_h_cropped, fov_v=fov_v_cropped) #720, 1280(
        # rgb_cam_mat = get_sim_cam_mat_with_fov2(rgb.shape[0], rgb.shape[1])   # no fov
        # feat_cam_mat = get_sim_cam_mat(features.shape[2], features.shape[3])  # no fov

        # Step3. project all the global point cloud onto the ground                      => 여기서 쓰이는 local과 global point cloud의 쓰임을 다시 확인할 것
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(args['gs'], args['cs'], p[0], p[2])

            # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            # Step4. rgb embedding vector
            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            # rgb_v = rgb[rgb_py, rgb_px, :]
            # semantic_v = semantic[rgb_py, rgb_px]
            # if semantic_v == 40:
            #     semantic_v = -1

            # when the projected location is already assigned a color value before, overwrite if the current point has larger height
            if 0 <= rgb_px < rgb.shape[1] and 0 <= rgb_py < rgb.shape[0]:  # 범위 검사를 여기서 수행
                rgb_v = rgb[rgb_py, rgb_px, :]
                if p_local[1] < color_top_down_height[y, x]:
                    color_top_down[y, x] = rgb_v
                    color_top_down_height[y, x] = p_local[1]
            
            # if p_local[1] < color_top_down_height[y, x]:
            #     color_top_down[y, x] = rgb_v
            #     color_top_down_height[y, x] = p_local[1]
            #     # gt[y, x] = semantic_v

            # average the visual embeddings if multiple points are projected to the same grid cell
            px, py, pz = project_point(feat_cam_mat, p_local)
            if 0 <= px < features.shape[3] and 0 <= py < features.shape[2]:  # 범위 검사를 여기서 수행
                feat = features[0, :, py, px]
                grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
                weight[y, x] += 1
            # if not (px < 0 or py < 0 or px >= features.shape[3] or py >= features.shape[2]):
            #     feat = features[0, :, py, px]
            #     grid[y, x] = (grid[y, x] * weight[y, x] + feat) / (weight[y, x] + 1)
            #     weight[y, x] += 1

            # build an obstacle map ignoring points on the floor (0 means occupied, 1 means free)
            if p_local[1] > args['camera_height']:
                continue
            obstacles[y, x] = 0
        pbar.update(1)
        
    save_map(color_top_down_save_path, color_top_down)
    # save_map(gt_save_path, gt)
    save_map(grid_save_path, grid)
    save_map(weight_save_path, weight)
    save_map(obstacles_save_path, obstacles)
