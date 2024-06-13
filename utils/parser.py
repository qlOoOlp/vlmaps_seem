import os, sys
import json
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # data path
    parser.add_argument('--root-path', type=str)
    parser.add_argument('--data-option', type=str, default='habitat_sim', choices=['habitat_sim', 'rtabmap'])
    parser.add_argument('--data-name', type=str, default='5LpN3gDmAk7_1')
    
    # related to vlmaps
    parser.add_argument('--cs', type=int, default=0.025) # 0.05
    parser.add_argument('--gs', type=int, default=2000)  # 1000
    parser.add_argument('--camera-height', type=float, default=1.5)
    parser.add_argument('--depth-sample-rate', type=int, default=100) # 50
    parser.add_argument('--mask-version', type=int, default=1)
    
    parser.add_argument('--pose', type=str, default='robot', choices=['robot', 'camera', 'scan'])

    # related to lseg
    parser.add_argument('--lseg-ckpt', type=str, default='/checkpoints/lseg/demo_e200.ckpt')
    parser.add_argument('--crop-size', type=int, default=480)
    parser.add_argument('--base-size', type=int, default=520)
    parser.add_argument('--lang', type=str, default='door,chair,ground,ceiling,other')

    # related to clip
    parser.add_argument('--clip-version', type=str, default='ViT-B/32', choices=['ViT-B/32', 'RN101'])

    args = parser.parse_args()
    print(args)

    if args.data_option=='habitat_sim':
        save_args(args)

    return args

def parse_args_load_map():
    parser = argparse.ArgumentParser()
    
    # data
    parser.add_argument('--root-path', type=str, default='/data')
    parser.add_argument('--data-option', type=str, default='habitat_sim', choices=['habitat_sim', 'rtabmap'])
    parser.add_argument('--data-name', type=str, default='5LpN3gDmAk7_1')

    # maps
    parser.add_argument('--color-map', action='store_true', help='if given, show top-down color map')
    parser.add_argument('--obstacle-map', action='store_true', help='if given, show obstacle map')
    parser.add_argument('--index-map', action='store_true', help='if given, show landmark indexing map')
    parser.add_argument('--index-option', type=str, default='mp3dcat', choices=['mp3dcat', 'lang'])
    
    parser.add_argument('--mask-version', type=int, default=1)

    args = parser.parse_args()
    print(args)

    return args

def save_args(args):
    data_path = os.path.join(args.root_path, args.data_option)

    if args.data_option=='habitat_sim':
        args.img_save_dir = os.path.join(data_path, args.data_name)
    elif args.data_option=='rtabmap':
        sub_save_dir = os.path.join(data_path, args.date)
        if len(os.listdir(sub_save_dir))==1:
            sub_option = os.listdir(sub_save_dir)[0]
        else:
            print("Here are sub options.")
            [print(f"Option{i+1}: {sub_option}") for i, sub_option in enumerate(os.listdir(sub_save_dir))]
            sub_option = input("Input sub option: ")
            
        args.img_save_dir = os.path.join(sub_save_dir, sub_option)

        if not os.path.exists(args.img_save_dir):
            print("제시된 Option을 재확인해주세요.")
            sys.exit(1)

    param_save_dir = os.path.join(args.img_save_dir, 'param')
    if not os.path.exists(param_save_dir):
        os.mkdir(param_save_dir)

    sub_save_dir = max([0]+[int(e) for e in os.listdir(param_save_dir)])+1
    param_sub_save_dir = os.path.join(param_save_dir, str(sub_save_dir))
    os.makedirs(param_sub_save_dir)

    with open(os.path.join(param_sub_save_dir, 'hparam.json'), 'w') as f:
        write_args = args.__dict__.copy()
        del write_args['device']
        json.dump(write_args, f, indent=4)