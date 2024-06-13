import os, sys

from vlmaps.create_maps import update_maps_sim, update_maps

from utils.parser import parse_args, save_args
from utils.clip_utils import get_clip_feats_dim
from utils.mapping_utils import initalize_maps

def main():
    # Step0. parameter 설정
    args = parse_args()
    mask_version = args.mask_version

    # Step1. path 설정
    data_path = os.path.join(args.root_path, args.data_option)
    if args.data_option=='rtabmap':
        try:
            print("Here is date that data is created.")
            [print(f"Option{i+1}: {date}") for i, date in enumerate(os.listdir(data_path))]
            if len(os.listdir(data_path))==1:
                args.date = os.listdir(data_path)[0]
            else: args.date = input("Input date: ")
            save_args(args)
        except FileNotFoundError:
            print("제시된 Option을 재확인해주세요.")
            sys.exit(1)

    img_save_dir = args.img_save_dir
    map_save_dir = os.path.join(img_save_dir, "map")
    if not os.path.exists(map_save_dir):
        os.mkdir(map_save_dir)

    param_save_dir = os.path.join(img_save_dir, 'param')
    param_save_path = os.path.join(param_save_dir, str(len(os.listdir(param_save_dir))), 'hparam.json')

    # device
    print(args.device)

    # clip
    clip_feat_dims = get_clip_feats_dim(args.clip_version)

    # rgb, depth, pose
    try:
        print(f"loading scene {img_save_dir}")
        rgb_dir = os.path.join(img_save_dir, "rgb")
        depth_dir = os.path.join(img_save_dir, "depth")
        pose_dir = os.path.join(img_save_dir, "pose")

        # rgb
        rgb_list = sorted(os.listdir(rgb_dir), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]

        # depth
        depth_list = sorted(os.listdir(depth_dir), key=lambda x: int(
            x.split("_")[-1].split(".")[0]))
        depth_list = [os.path.join(depth_dir, x) for x in depth_list]
        
        if args.data_option=='habitat_sim':
            pose_list = sorted(os.listdir(pose_dir), key=lambda x: int(
                x.split("_")[-1].split(".")[0]))
            pose_list = [os.path.join(pose_dir, x) for x in pose_list]

            lists = [rgb_list, depth_list, pose_list]

        elif args.data_option=='rtabmap':
            lists = [rgb_list, depth_list]

    except FileNotFoundError:
        print("rgb_dir, depth_dir, pose_dir 경로를 재확인해주세요.")
        print(f"rgb_dir: {rgb_dir}")
        print(f"depth_dir: {depth_dir}")
        sys.exit(1)

    # Step1. set maps
    # save dir
    color_top_down_save_path = os.path.join(map_save_dir, f"color_top_down_{mask_version}.npy")
    # gt_save_path = os.path.join(map_save_dir, f"grid_{mask_version}_gt.npy")
    grid_save_path = os.path.join(map_save_dir, f"grid_lseg_{mask_version}.npy")
    weight_save_path = os.path.join(map_save_dir, f"weight_lseg_{mask_version}.npy")
    obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

    save_paths = [color_top_down_save_path, grid_save_path, weight_save_path, obstacles_save_path, param_save_path]
    color_top_down_height = initalize_maps(save_paths, args.gs, args.camera_height, clip_feat_dims)

    # Step2. update maps
    if args.data_option=='habitat_sim':
        update_maps_sim(save_paths, lists, color_top_down_height)
    elif args.data_option=='rtabmap':
        update_maps(save_paths, lists, color_top_down_height)

if __name__=="__main__":
    main()