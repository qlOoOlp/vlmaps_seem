import os, json
import torch
import cv2
import numpy as np
from tqdm import tqdm

from utils.lseg_utils import get_lseg_feat
from utils.mapping_utils import save_map, load_pose, load_map, depth2pc, transform_pc, get_sim_cam_mat, pos2grid_id, project_point, load_depth_npy
from utils.preprocess import png_to_npy
from utils.get_transform import get_transform

from lseg.modules.models.lseg_net import LSegEncNet

def update_maps_sim(save_paths, lists, color_top_down_height):
    # load paths & lists
    color_top_down_save_path, grid_save_path, weight_save_path, obstacles_save_path, param_save_path = save_paths

    color_top_down = load_map(color_top_down_save_path)
    grid = load_map(grid_save_path)
    weight = load_map(weight_save_path)
    obstacles = load_map(obstacles_save_path)

    # data
    rgb_list, depth_list, pose_list = lists

    # read hparams
    with open(param_save_path, 'r') as f:
        args = json.load(f)

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

    # load & transform
    tf_list = []
    data_iter = zip(rgb_list, depth_list, pose_list)
    pbar = tqdm(total=len(rgb_list))

    # load all images and depths and poses per one image
    for data_sample in data_iter:
        rgb_path, depth_path, pose_path = data_sample

        # 1. read rgb
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

        # 2. read depth
        depth = load_depth_npy(depth_path)

        # 3. read semantic
        # semantic = load_semantic_npy(semantic_path)
        # semantic = cvt_obj_id_2_cls_id(semantic, obj2cls)

        # lseg features
        labels = args['lang'].split(",")
        transform, _MEAN, _STD = get_transform()

        pix_feats = get_lseg_feat(model, rgb, labels, transform, args['crop_size'], args['base_size'], _MEAN, _STD)

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

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])               # no fov
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])  # no fov

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
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
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
    # Step 0.
    # load paths & lists
    color_top_down_save_path, grid_save_path, weight_save_path, obstacles_save_path, param_save_path = save_paths

    color_top_down = load_map(color_top_down_save_path)
    grid = load_map(grid_save_path)
    weight = load_map(weight_save_path)
    obstacles = load_map(obstacles_save_path)

    # data
    rgb_list, depth_list = lists

    # read hparams
    with open(param_save_path, 'r') as f:
        args = json.load(f)

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

    # load & transform
    tf_list = []
    pose_dir = os.path.join(args['img_save_dir'], "pose")
    pose_txt = os.path.join(pose_dir, f"{args['pose']}.txt")
    f = open(pose_txt, "r")
    f.readline()
    pbar = tqdm(total=len(rgb_list))

    # load all images and depths and poses per one image
    for data_sample in zip(rgb_list, depth_list, f.readlines()):
        rgb_path, depth_path, pose_path = data_sample

        # 1. read rgb
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 2. read pose: transform to camera coordinate
        pos, rot = load_pose(pose_path, flag=False)
        rot_ro_cam = np.eye(3)              
        rot = rot @ rot_ro_cam

        pos[1] += (args['camera_height'])

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)
        if len(tf_list) == 1:
            init_tf_inv = np.linalg.inv(tf_list[0])

        tf = init_tf_inv @ pose

        # 3. read depth
        depth = png_to_npy(depth_path)

        # lseg features
        labels = args['lang'].split(",")
        transform, _MEAN, _STD = get_transform()
        pix_feats = get_lseg_feat(model, rgb, labels, transform, args['crop_size'], args['base_size'], _MEAN, _STD)

        # Step1. depth to local point cloud
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

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])             
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        # Step3. project all the global point cloud onto the ground
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            x, y = pos2grid_id(args['gs'], args['cs'], p[0], p[2])

            # ignore points projected to outside of the map and points that are 0.5 higher than the camera (could be from the ceiling)
            if x >= obstacles.shape[0] or y >= obstacles.shape[1] or \
                x < 0 or y < 0 or p_local[1] < -0.5:
                continue

            # Step4. rgb embedding vector
            # when the projected location is already assigned a color value before, overwrite if the current point has larger height
            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]
            if 0 <= rgb_px < rgb.shape[1] and 0 <= rgb_py < rgb.shape[0]:  # 범위 검사를 여기서 수행
                if p_local[1] < color_top_down_height[y, x]:
                    color_top_down[y, x] = rgb_v
                    color_top_down_height[y, x] = p_local[1]

            # average the visual embeddings if multiple points are projected to the same grid cell
            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
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