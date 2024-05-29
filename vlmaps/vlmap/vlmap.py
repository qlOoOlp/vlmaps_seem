
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import clip
from utils.mapping_utils import load_map, get_new_pallete, get_new_mask_pallete
from utils.clip_utils import get_text_feats

from utils.mp3dcat import mp3dcat
from utils.lang import langs

def show_color_map(mm_values, path):
    xmin, xmax, ymin, ymax = mm_values
    color_top_down = load_map(path)
    
    color_top_down = color_top_down[xmin:xmax+1, ymin:ymax+1]
    color_top_down_pil = Image.fromarray(color_top_down)

    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(color_top_down_pil)
    plt.show()

def show_obstacle_map(mm_values, path):
    xmin, xmax, ymin, ymax = mm_values
    obstacles = load_map(path)

    print(np.unique(obstacles))

    obstacles_pil = Image.fromarray(obstacles[xmin:xmax+1, ymin:ymax+1])
    plt.figure(figsize=(8, 6), dpi=120)
    plt.imshow(obstacles_pil, cmap='gray')
    plt.show()

def show_index_map(opt, no_map_mask, mm_values, path):
    xmin, xmax, ymin, ymax = mm_values

    # index
    if opt=='mp3dcat': lang = mp3dcat
    elif opt=='lang': lang = langs
    
    # clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()

    text_feats = get_text_feats(lang, clip_model, clip_feat_dim)

    grid = load_map(path)
    grid = grid[xmin:xmax+1, ymin:ymax+1]
    map_feats = grid.reshape((-1, grid.shape[-1]))
    scores_list = map_feats @ text_feats.T

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
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(handles=patches, loc='upper left', bbox_to_anchor=(1., 1), prop={'size': 10})
    plt.axis('off')
    plt.title("VLMaps")
    plt.imshow(seg)
    plt.show()