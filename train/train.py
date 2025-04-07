"""
TODO: 使用dinov2预训练模型进行微调,使用Adapter Fine-Tuning的方式
      进行小参量微调, 同时记录模型运行过程debug
Time: 2025/03/14-Redal
"""
import os
import sys
import argparse
import h5py
import pathlib

import torch
import torchvision
from torchvision.transforms import transform



#######################  定义参量阈  ###########################

