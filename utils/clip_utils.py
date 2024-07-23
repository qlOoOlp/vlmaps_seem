import numpy as np
import torch
import clip
import open_clip

def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64, clip_version=None, is_openclip=False):
    if not is_openclip:
        if torch.cuda.is_available():
            # 1. tokenize 진행
            text_tokens = clip.tokenize(in_text).cuda()
        # elif torch.backends.mps.is_available():
        #     text_tokens = clip.tokenize(in_text).to("mps")
        text_id = 0
        text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)  # (len(in_text), 512)
        
        while text_id < len(text_tokens):  # Batched inference.
            batch_size = min(len(in_text) - text_id, batch_size)    
            text_batch = text_tokens[text_id : text_id + batch_size]
            with torch.no_grad():
                batch_feats = clip_model.encode_text(text_batch).float()
            batch_feats /= batch_feats.norm(dim=-1, keepdim=True) # 정규화
            batch_feats = np.float32(batch_feats.cpu())
            text_feats[text_id : text_id + batch_size, :] = batch_feats
            text_id += batch_size
    else:
        if '/' in clip_version: clip_version = clip_version.replace("/","-")
        tokenizer = open_clip.get_tokenizer(clip_version)
        text = tokenizer(in_text)

        # BATCH 없는 버전
        text_feats = clip_model.encode_text(text.cuda())
        text_feats = torch.nn.functional.normalize(text_feats, dim=-1)  # [len(in_text), feat_dim]
    print(f"text_feats: {text_feats.shape}")
    return text_feats

def get_clip_feat_dims(version):
    clip_versions = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN0x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768,
                    'ViT-H/14': 1024}
    return clip_versions[version]