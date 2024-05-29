import numpy as np
import torch
import clip

def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    if torch.cuda.is_available():
        text_tokens = clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)  # (len(in_text), 512)
    
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)    
        text_batch = text_tokens[text_id : text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id : text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats

def get_clip_feats_dim(clip_version):
    clip_versions = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN0x16': 768,
                'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}
    return clip_versions[clip_version]