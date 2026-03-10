import argparse
import time
from pathlib import Path
import sys
from ctypes import *
import numpy as np
import cv2
import torch


def run(
        status_label=None,
):
    torch.cuda.empty_cache()
