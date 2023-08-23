from swinFSR_convMLP_DCA_SFA import SwinFSR
import logging
import os
import tempfile
import shutil
import sys
from abc import ABC, abstractmethod
from monai.utils import first, set_determinism
import matplotlib.pyplot as plt
import SimpleITK as sitk  # noqa: N813
import numpy as np
import itk
from PIL import Image
import tempfile
from monai.data import ITKReader, PILReader
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import PSNRMetric, SSIMMetric
from monai.losses import DiceLoss
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from monai.apps import CrossValidation
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, create_test_image_3d
from monai.engines import (
  EnsembleEvaluator,
  SupervisedEvaluator,
  SupervisedTrainer
)
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer, sliding_window_inference

from monai.losses import DiceLoss
from monai.networks.nets import UNet
import torch
from monai.transforms import (
  AsDiscrete,
  AsDiscreted,
  Activationsd,
  EnsureChannelFirstd,
  AsChannelLast,
  AsDiscreted,
  Compose,
  Resized,
  LoadImaged,
  LoadImage,
  MeanEnsembled,
  RandCropByPosNegLabeld,
  RandRotate90d,
  RandCropByPosNegLabeld,
  SaveImaged,
  Orientationd,
  ScaleIntensityRanged,
  Spacingd,
  EnsureTyped,
  VoteEnsembled,
  Flipd,
  SavitzkyGolaySmoothd,
  MedianSmoothd,
  ResizeWithPadOrCropd,
  ScaleIntensityRange,
  Transposed,
  RandFlipd,
  GaussianSmoothd,
  RandRotate90d,
  RandShiftIntensityd,
  ToTensor,
)
from monai.utils import set_determinism
%matplotlib inline
print_config()

os.environ['MONAI_DATA_DIRECTORY'] = 'Path/to/weights'
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

import glob
train_image_list = glob.glob('train/data')
train_label_list = train_image_list.copy()
len(train_image_list)

test_image_list = glob.glob('test/data')
test_label_list = test_image_list.copy()
len(test_image_list)

import monai
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

def transform():
  train_transforms = Compose(
      [
          LoadImaged(keys=["image", "label"]),
          EnsureChannelFirstd(keys=["image," "label"]),
          Transposed(keys=["image", "label"],indices=(0,2,1)),
          Resized(keys=["image"],spatial_size=(256, 256), size_mode='all', mode='bicubic', align_corners=None, anti_aliasing=True, anti_aliasing_sigma=None),
          #Resized(keys=["image"],spatial_size=(128, 128, 3), size_mode='all', mode='area', align_corners=None, anti_aliasing=False, anti_aliasing_sigma=None),
          ScaleIntensityRanged(
              keys=["image","label"], a_min=0.0, a_max=255.0,
              b_min=0.0, b_max=1.0, clip=True,
          ),
      ]
  )




def train():
  upscale = 2
  window_size = 8
  height = (512 // upscale // window_size + 1) * window_size
  width = (512 // upscale // window_size + 1) * window_size
  print(height,width)
  model = SwinFSR(upscale=upscale, img_size=(height, width),
               window_size=window_size, img_range=1., depths=[4, 4, 4, 4],
               embed_dim=60, num_heads=[6, 6, 6, 6], sfa_blocks=4,mlp_ratio=2, upsampler='pixelshuffledirect').to(device)

  loss_function =  torch.nn.L1Loss()#DiceLoss(to_onehot_y=True, softmax=True)
  optimizer = torch.optim.Adam(model.parameters(), 2e-4)
  psnr_metric = PSNRMetric(max_val=255,reduction="mean")
  
  d_range = torch.ones(1)*255
  ssim_metric=  SSIMMetric(data_range=d_range.to(device),reduction="mean")
  
  

