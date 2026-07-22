"""Convert training checkpoint to serialized model for run_demo.py"""
import torch
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Monkey-patch timm.create_model to force pretrained=False
import timm
_original_create = timm.create_model
def _patched_create(*args, **kwargs):
    kwargs['pretrained'] = False
    return _original_create(*args, **kwargs)
timm.create_model = _patched_create

from omegaconf import OmegaConf
from core.foundation_stereo import FastFoundationStereo

ckpt_path = "weights/23-36-37/model_best_bp2.pth"
cfg_path = "weights/23-36-37/cfg.yaml"
out_path = "weights/23-36-37/model_best_bp2_serialize.pth"

print(f"Loading checkpoint from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print(f"Loading config from {cfg_path}...")
with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)
args = OmegaConf.create(cfg)

print("Building model (pretrained=False)...")
model = FastFoundationStereo(args)

print("Loading state dict...")
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
if missing:
    print(f"  Missing keys: {len(missing)}")
    for k in missing[:5]:
        print(f"    {k}")
if unexpected:
    print(f"  Unexpected keys: {len(unexpected)}")
    for k in unexpected[:5]:
        print(f"    {k}")

print(f"Saving serialized model to {out_path}...")
torch.save(model, out_path)
sz = os.path.getsize(out_path) / (1024**3)
print(f"Done! Saved to {out_path} ({sz:.1f} GB)")
