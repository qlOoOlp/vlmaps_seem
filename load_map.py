import sys, os
import torch
import numpy as np

from vlmaps.vlmap.vlmap import show_color_map, show_obstacle_map, show_index_map

from utils.parser import parse_args_load_map
from utils.mapping_utils import load_map
from utils.clip_utils import get_clip_feat_dims

import clip, open_clip

def main():
    # Step0. parameter 설정
    args = parse_args_load_map()
    mask_version = args.mask_version

    # Step1. path 설정
    data_path = os.path.join(args.root_path, args.data_option)
    if args.data_option=='habitat_sim':
        img_save_dir = os.path.join(data_path, args.data_name)
    elif args.data_option=='rtabmap':
        try:
            print("Here is date that data is created.")
            [print(f"Option{i+1}: {date}") for i, date in enumerate(os.listdir(data_path))]
            if len(os.listdir(data_path))==1:
                date = os.listdir(data_path)[0]
            else: date = input("Input date: ")

            print("Here are sub options.")
            sub_save_dir = os.path.join(data_path, date)
            [print(f"Option{i+1}: {sub_option}") for i, sub_option in enumerate(os.listdir(sub_save_dir))]
            if len(os.listdir(sub_save_dir))==1:
                sub_option = os.listdir(sub_save_dir)[0]
            else:          
                sub_option = input("Input sub option: ")
            img_save_dir = os.path.join(sub_save_dir, sub_option)
        except FileNotFoundError:
            print("제시된 Option을 재확인해주세요.")
            sys.exit(1)

    args.map_save_dir = os.path.join(img_save_dir, "map_lseg")   # "map_cf"
    map_save_dir = args.map_save_dir

    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    # gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    # extract min & max values
    obstacles = load_map(obstacles_save_path)
    x_indices, y_indices = np.where(obstacles == 0)
    mm_values = [np.min(x_indices), np.max(x_indices), np.min(y_indices), np.max(y_indices)]

    # show maps
    if not os.path.exists(args.img_save_dir):
        os.mkdir(args.img_save_dir)
    sub_save_dir = max([0]+[int(e) for e in os.listdir(args.img_save_dir)])+1
    map_sub_save_dir = os.path.join(args.img_save_dir, str(sub_save_dir))
    os.makedirs(map_sub_save_dir)

    if args.obstacle_map: show_obstacle_map(mm_values, [obstacles_save_path, map_sub_save_dir])
    if args.color_map: show_color_map(mm_values, [color_top_down_save_path, map_sub_save_dir])
    if args.index_map: 
        no_map_mask = obstacles[mm_values[0]:mm_values[1]+1, mm_values[2]:mm_values[3]+1] > 0
        # obstacles_rgb = np.repeat(obstacles[mm_values[0]:mm_values[1]+1, mm_values[2]:mm_values[3]+1, None], 3, axis=2)
        print(no_map_mask.shape)
        clip_feat_dim = get_clip_feat_dims(args.clip_version)
        print(clip_feat_dim)
        
        if not args.openclip:
            # clip
            clip_model, _ = clip.load(args.clip_version)  # clip.available_models()
            clip_model.cuda().eval()
            # show_index_map(args.index_option, no_map_mask, mm_values, [grid_save_path, map_sub_save_dir])
        else:
            # OpenCLIP
            print(
                f"Initializing OpenCLIP model: {args.clip_version}"
                f" pre-trained on {args.openclip_pretrained_dataset}..."
            )
            clip_model, _, _ = open_clip.create_model_and_transforms(
                args.clip_version, args.openclip_pretrained_dataset
            )
            clip_model.cuda().eval()
        show_index_map(args.index_option, no_map_mask, mm_values, [grid_save_path, map_sub_save_dir], \
                            clip_model, clip_feat_dim, args.clip_version, args.openclip)

if __name__ == "__main__":
    main()