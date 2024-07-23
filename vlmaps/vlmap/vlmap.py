import os

import torch
import numpy as np
import open_clip
import matplotlib.pyplot as plt
from PIL import Image

import clip
from utils.mapping_utils import load_map, get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats# get_text_feats, get_clip_feat_dims, get_text_feats_cf,  
# from utils.extract_conceptfusion_features_vlmaps import input_text_query

from utils.mp3dcat import mp3dcat
from utils.lang import langs

def show_color_map(mm_values, paths):
    color_map_path, map_save_dir = paths
    save_path = os.path.join(map_save_dir, "color_map.png")

    xmin, xmax, ymin, ymax = mm_values
    color_top_down = load_map(color_map_path)
    
    color_top_down = color_top_down[xmin:xmax+1, ymin:ymax+1]
    color_top_down_pil = Image.fromarray(color_top_down)
    
    color_top_down_pil.save(save_path)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(color_top_down_pil)
    plt.show()

def show_obstacle_map(mm_values, paths):
    obstacle_map_path, map_save_dir = paths
    save_path = os.path.join(map_save_dir, "obs_map.png")

    xmin, xmax, ymin, ymax = mm_values
    obstacles = load_map(obstacle_map_path)
    # print(np.unique(obstacles))
    
    obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
    obstacles_pil.putpalette([0,0,0,255,255,255])
    obstacles_pil.save(save_path)
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(obstacles_pil, cmap='gray')
    plt.show()

def show_index_map(opt, no_map_mask, mm_values, paths, clip_model, clip_feat_dim, clip_version, is_openclip):
    idx_map_path, map_save_dir = paths
    save_path = os.path.join(map_save_dir, "vlmap.png")
    xmin, xmax, ymin, ymax = mm_values

    # index
    if opt=='mp3dcat': lang = mp3dcat
    elif opt=='lang': lang = langs

    # grid map
    grid = load_map(idx_map_path)
    grid = grid[xmin:xmax+1, ymin:ymax+1]           # (?, ?, 512)
    map_feats = grid.reshape((-1, grid.shape[-1]))  # (?*?, 512)

    ###########
    if not is_openclip:
        text_feats = get_text_feats(lang, clip_model, clip_feat_dim)            # (len(lang), clip_feat_dim)
        scores_list = map_feats @ text_feats.T                                  # (?*?, len(lang))
    else:
        text_feats = get_text_feats(lang, clip_model, clip_feat_dim, clip_version=clip_version, is_openclip=is_openclip)
        text_feats = text_feats.cpu().detach().numpy()
        scores_list = map_feats @ text_feats.T                                  # (?*?, len(lang))
    ###########

    predicts = np.argmax(scores_list, axis=1)
    predicts = predicts.reshape((xmax - xmin + 1, ymax - ymin + 1))
    floor_mask = predicts == 2

    new_pallete = get_new_pallete(len(lang))
    mask, patches = get_new_mask_pallete(predicts, new_pallete, out_label_flag=True, labels=lang)
    seg = mask.convert("RGBA")
    seg = np.array(seg)

    seg[no_map_mask] = [225, 225, 225, 255]
    seg[floor_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)

    seg.save(save_path)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    plt.title("VLMaps")
    plt.imshow(seg)
    plt.show()