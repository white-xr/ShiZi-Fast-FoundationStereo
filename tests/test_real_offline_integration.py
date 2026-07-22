from pathlib import Path

import pytest
import yaml


def test_real_stereo_pair_locator_integration():
  repo = Path(__file__).resolve().parents[1]
  calibration = yaml.safe_load((repo / 'configs/stereo_calib.yaml').read_text(encoding='utf-8'))
  model = repo / 'weights/23-36-37/model_best_bp2_serialize.pth'
  yolo = repo / 'weights/triangle-metal.pt'
  pair = repo / 'workspace/data/A相机→白板268/left_rgb/000011.png'
  if not calibration.get('configured', False):
    pytest.skip('real Gemini dual-RGB stereo calibration is not configured')
  if not model.exists() or not yolo.exists() or not pair.exists():
    pytest.skip('real model weights or captured stereo pair are unavailable')
  pytest.skip('set up a dedicated hardware-safe RUN_REAL_FFS_INTEGRATION job before executing GPU integration')
