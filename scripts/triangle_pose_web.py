#!/usr/bin/env python3
import sys
from pathlib import Path


repo_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_dir))

from scripts.screw_offset_web import main


if __name__ == '__main__':
  main()
