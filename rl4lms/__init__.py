import os
import torch
import numpy as np
from stable_baselines3.common.utils import conv_bfloat16

aapd_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/AAPD" )
