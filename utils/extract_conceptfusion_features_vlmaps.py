#%%
import os
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
import open_clip
import torch
import torch.nn.functional as F
import tyro
# from gradslam.datasets import (
#     ICLDataset,
#     ReplicaDataset,
#     ScannetDataset,
#     load_dataset_config,
# )
# from gradslam.slam.pointfusion import PointFusion
# from gradslam.structures.pointclouds import Pointclouds
# from gradslam.structures.rgbdimages import RGBDImages
# from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image
# from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm, trange
from typing_extensions import Literal

@dataclass
class ProgramArgs:
    # Torch device to run computation on (E.g., "cpu")
    device: str = "cuda"

    # SAM checkpoint and model params
    root_dir: str = "/home/home/0710/grounded-sam-note"
    checkpoint_path: str = os.path.join(root_dir, "checkpoints", "sam_vit_b_01ec64.pth") # "sam_vit_h_4b8939.pth", "sam_vit_l_0b3195.pth")
    model_type = "vit_b"
    # Ignore masks that have valid pixels less than this fraction (of the image area)
    bbox_area_thresh: float = 0.0005
    # Number of query points (grid size) to be sampled by SAM
    points_per_side: int = 32

    # CLIP model config
    # open_clip_model = "ViT-B-16" # "ViT-L-14" # "ViT-H-14"
    # open_clip_pretrained_dataset = "laion2b_s34b_b88k" # "laion2b_s32b_b82k" # "laion2b_s32b_b79k"

    # Desired image width and height
    desired_height: int = 347
    desired_width: int = 520

    # Directory to save extracted features
    save_dir: str = "saved-feat"

    ###### 
    mask_version: int = 1
    openclip: bool = True
    clip_version: str = 'ViT-B/16'

    data_option: str = "rtabmap"
    depth_sample_rate: int = 10
    camera_height: float = 0.5
    ###### 

def resize_image(img, h, w, **up_kwargs):
    return F.interpolate(img, (h, w), **up_kwargs)

def extract_conceptfusion(mask_generator, openclip, preprocess, img_path):

    torch.autograd.set_grad_enabled(False)
    args = tyro.cli(ProgramArgs)
    os.makedirs(args.save_dir, exist_ok=True)

    # Step0. load a image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = img.shape[0], img.shape[1] 
    
    # Step1. Using SAM 
    masks = mask_generator.generate(img)

    # Step2. Calculate pixel-aligned features    
    # Step2-1. Extract a global feature (CLIP)
    global_feat = None
    with torch.cuda.amp.autocast():
        # print("Extracting global CLIP features...")
        _img = preprocess(Image.open(img_path)).unsqueeze(0)
        global_feat = openclip.encode_image(_img.cuda())                 # --> (1, 1024)
        global_feat /= global_feat.norm(dim=-1, keepdim=True)            # 한 행에 대해 normalize   
        # tqdm.write(f"Image feature dims: {global_feat.shape} \n")
    global_feat = global_feat.half().cuda()
    global_feat = torch.nn.functional.normalize(global_feat, dim=-1)     # --> (1, 1024)
    feat_dim = global_feat.shape[-1]

    # Step2-2. Extract local(= region) features (SAM)
    feat_per_roi = []
    roi_nonzero_inds = []
    similarity_scores = []
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    for maskidx in range(len(masks)):
        _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
        _x, _y, _w, _h = int(_x), int(_y), int(_w), int(_h)

        seg = masks[maskidx]["segmentation"]
        nonzero_inds = torch.argwhere(torch.from_numpy(seg))
        # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
        img_roi = img[_y : _y + _h, _x : _x + _w, :]
        img_roi = Image.fromarray(img_roi)
        img_roi = preprocess(img_roi).unsqueeze(0).cuda()   # img_roi: (42, 97, 3), preprocess(img_roi): (3, 224, 224)

        roifeat = openclip.encode_image(img_roi)
        roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
        feat_per_roi.append(roifeat)
        roi_nonzero_inds.append(nonzero_inds)

        _sim = cosine_similarity(global_feat, roifeat)
        similarity_scores.append(_sim)
    
    # Step3-3. Fuse pixels
    similarity_scores = torch.cat(similarity_scores)
    softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
    outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.half)
    for maskidx in range(len(masks)):
        _weighted_feat = softmax_scores[maskidx] * global_feat + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
        _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
        outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] += _weighted_feat[0].detach().cpu().half()
        outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]] = torch.nn.functional.normalize(
            outfeat[roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]].float(), dim=-1
        ).half()

    outfeat = outfeat.unsqueeze(0).float()  # interpolate is not implemented for float yet in pytorch
    outfeat = outfeat.permute(0, 3, 1, 2)   # 1, H, W, feat_dim -> 1, feat_dim, H, W
    outfeat = torch.nn.functional.interpolate(outfeat, [args.desired_height, args.desired_width], mode="nearest")
    outfeat = outfeat.permute(0, 2, 3, 1)   # 1, feat_dim, H, W --> 1, H, W, feat_dim
    outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
    outfeat = outfeat.permute(0, 3, 1, 2)

    return outfeat.cpu().numpy() # (1, feat_dim(1024), H(120), W(160)) 
    # return outfeat.half().cpu().numpy() # (1, feat_dim(1024), H(120), W(160)) 

def input_text_query(outfeat, prompt_text_list, img_path):
    '''
    outfeat.shape = (H(120), W(160), feat_dim(1024)) --> (H*W, feat_dim) --> (H*W, 1, feat_dim)
    textfeat.shape = (len(prompt_text_list), feat_dim(1024))             --> (1, len(prompt_text_list), feat_dim)
    '''
    args = tyro.cli(ProgramArgs)
    
    feat_dim = outfeat.shape[-1]
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = img.shape[0], img.shape[1] 

    # Step1. Load CLIP model
    print(
        f"Initializing OpenCLIP model: {args.clip_verion}"
        f" pre-trained on {args.open_clip_pretrained_dataset}..."
    )
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_verion, args.open_clip_pretrained_dataset
    )
    # model.cuda()
    model.eval()
    
    # Step2. Extract text embeddings
    if '/' in args.clip_version: clip_version = clip_version.replace("/","-")
    tokenizer = open_clip.get_tokenizer(clip_version)
    text = tokenizer(prompt_text_list)
    textfeat = model.encode_text(text.cuda())
    textfeat = torch.nn.functional.normalize(textfeat, dim=-1)  # [len(prompt_text_list), feat_dim]
    print(f"textfeat: {textfeat.shape}")

    # Step3. Calculate embeddings' correlation
    outfeat = outfeat.reshape(-1, feat_dim)                     # [H*W, feat_dim]
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    similarity = cosine_similarity(outfeat.unsqueeze(1).cuda(), textfeat.unsqueeze(0))
    similarity = similarity.reshape(args.desired_height, args.desired_width, len(prompt_text_list))
    similarity = similarity.unsqueeze(0)         # 1, H, W, feat_dim
    similarity = similarity.permute(0, 3, 1, 2)  # 1, H, W, feat_dim --> 1, feat_dim, H, W
    similarity = torch.nn.functional.interpolate(similarity, [LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH], mode="nearest")

    return similarity  # (B, num, H, W)
