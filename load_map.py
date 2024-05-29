import sys, os
import numpy as np

from vlmaps.vlmap.vlmap import show_color_map, show_obstacle_map, show_index_map

from utils.parser import parse_args_load_map, save_args
from utils.mapping_utils import load_map

def main():
    args = parse_args_load_map()
    mask_version = args.mask_version

    # Step1. habitat_sim data 의 경우
    data_path = os.path.join(args.root_path, args.data_option)
    if args.data_option=='habitat_sim':
        img_save_dir = os.path.join(data_path, '5LpN3gDmAk7_1')
    elif args.data_option=='rtabmap':
        try:
            print("Here is date that data is created.")
            [print(f"Option{i+1}: {date}") for i, date in enumerate(os.listdir(data_path))]
            date = input("Input date: ")
            
            print("Here are sub options.")
            sub_save_dir = os.path.join(data_path, date)
            [print(f"Option{i+1}: {sub_option}") for i, sub_option in enumerate(os.listdir(sub_save_dir))]
            sub_option = input("Input sub option: ")
            img_save_dir = os.path.join(sub_save_dir, sub_option)
        except FileNotFoundError:
            print("제시된 Option을 재확인해주세요.")
            sys.exit(1)

    args.map_save_dir = os.path.join(img_save_dir, "map")
    # save_args(args)

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
    if args.obstacle_map: show_obstacle_map(mm_values, obstacles_save_path)
    if args.color_map: show_color_map(mm_values, color_top_down_save_path)
    if args.index_map: 
        no_map_mask = obstacles[mm_values[0]:mm_values[1]+1, mm_values[2]:mm_values[3]+1] > 0
        # obstacles_rgb = np.repeat(obstacles[mm_values[0]:mm_values[1]+1, mm_values[2]:mm_values[3]+1, None], 3, axis=2)
        print(no_map_mask.shape)

        show_index_map(args.index_option, no_map_mask, mm_values, grid_save_path)

if __name__ == "__main__":
    main()