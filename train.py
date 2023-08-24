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
import monai
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

os.environ['MONAI_DATA_DIRECTORY'] = 'training_weights'
if not os.path.exists('training_weights'):
    os.makedirs('training_weights')
directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#device = torch.device("cuda" if torch.cuda.is_available() else "CPU")

def transform(spatial_size=(256,256),mode='bicubic',a_max=255.0,a_min=0.0,b_max=1.0,b_min=0.0):
    transforms = Compose(
      [
          LoadImaged(keys=["image", "label"]),
          EnsureChannelFirstd(keys=["image," "label"]),
          Transposed(keys=["image", "label"],indices=(0,2,1)),
          Resized(keys=["image"],spatial_size=spatial_size, size_mode='all', mode=mode, align_corners=None, anti_aliasing=True, anti_aliasing_sigma=None),
          ScaleIntensityRanged(
              keys=["image","label"], a_min=a_min, a_max=a_max,
              b_min=b_min, b_max=b_max, clip=True,
          ),
      ]
    )
    return transforms

def post_transform(a_min=0.0,a_max=1.0,b_min=0.0,b_max=255.0):
    post_transforms = Compose(
        [
            ScaleIntensityRange( a_min=a_min, a_max=a_max,
                b_min=b_min, b_max=b_max, clip=True,
            ),
            ToTensor(),
        ]
    )
    return post_transforms

def dataloader(train_path='data/SANS/Train_test_split/Train/*.png',test_path='data/SANS/Train_test_split/Test/*.png'):
    train_image_list = glob.glob(train_path)
    train_label_list = train_image_list.copy()
    
    test_image_list = glob.glob(test_path)
    test_label_list = test_image_list.copy()
    
    train_dicts = [{"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_image_list, train_label_list)]
    
    test_dicts = [{"image": image_name, "label": label_name}
        for image_name, label_name in zip(test_image_list, test_label_list)]
    
    return train_dicts, test_dicts

def train(image_size=(512,512),spatial_size=(256,256),
          train_batch_size=2,val_batch_size=1,train_path,test_path,upscale=2,window_size=8,depths=[4,4,4,4],
          sfa_blocks=4,embed_dim=60,num_heads=[6,6,6,6],lr=2e-4,epochs=200,val_interval=2):
          
        max_epochs = epochs
        val_interval = val_interval
        best_metric_psnr = -1
        best_metric_ssim = -1
        best_metric_psnr_epoch = -1
        best_metric_ssim_epoch = -1
        epoch_loss_values = []
        psnr_metric_values = []
        ssim_metric_values = []
        for epoch in range(max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        val_outputs= model(val_inputs)
                        val_outputs = [post_transforms(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_transforms(i) for i in decollate_batch(val_labels)]

                        psnr_metric(y_pred=val_outputs, y=val_labels)
                        ssim_metric(y_pred=val_outputs, y=val_labels)

                    psnr_metric_curr = psnr_metric.aggregate().item()
                    ssim_metric_curr = ssim_metric.aggregate().item()
                    psnr_metric.reset()
                    ssim_metric.reset()

                    psnr_metric_values.append(psnr_metric_curr)
                    ssim_metric_values.append(ssim_metric_curr)

                    if psnr_metric_curr > best_metric_psnr:
                        best_metric_psnr = psnr_metric_curr
                        best_metric_psnr_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(
                            root_dir, "best_psnr_model.pth"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean psnr: {psnr_metric_curr:.4f}"
                        f"\nbest mean psnr: {best_metric_psnr:.4f} "
                        f"at epoch: {best_metric_psnr_epoch}"
                    )

                    if ssim_metric_curr > best_metric_ssim:
                        best_metric_ssim = ssim_metric_curr
                        best_metric_ssim_epoch = epoch + 1
                        torch.save(model.state_dict(), os.path.join(
                            root_dir, "best_ssim_model.pth"))
                        print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current mean ssim: {ssim_metric_curr:.4f}"
                        f"\nbest mean ssim: {best_metric_ssim:.4f} "
                        f"at epoch: {best_metric_ssim_epoch}"
                    )





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--npz_file', type=str, default='vtgan', help='path/to/npz/file')
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--n_patch', type=int, default=64)
    parser.add_argument('--savedir', type=str, required=False, help='path/to/save_directory',default='VTGAN')
    parser.add_argument('--resume_training', type=str, required=False,  default='no', choices=['yes','no'])
    args = parser.parse_args()
    
    device = torch.device("cuda:0")
    train_dicts, test_dicts = dataloader(train_path,test_path)
    train_transforms = transform(spatial_size)
    train_ds = CacheDataset(data=train_dicts, transform=train_transforms,cache_rate=1.0, num_workers=2)
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=2)

    val_transforms= transform(patial_size)
    val_ds = CacheDataset(data=test_dicts, transform=val_transforms, cache_rate=1.0, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=2)

    upscale = upscale
    window_size = window_size
    height = (image_size.shape[0] // upscale // window_size + 1) * window_size
    width = (image_size.shape[1] // upscale // window_size + 1) * window_size
    print(height,width)
    model = SwinFSR(upscale=upscale, img_size=(height, width),
               window_size=window_size, img_range=1., depths=depths,
               embed_dim=embed_dim, num_heads=num_heads, sfa_blocks=sfa_blocks,mlp_ratio=2, upsampler='pixelshuffledirect').to(device)

    loss_function =  torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    psnr_metric = PSNRMetric(max_val=255,reduction="mean")

    d_range = torch.ones(1)*255
    ssim_metric=  SSIMMetric(data_range=d_range.to(device),reduction="mean")
      
        
    
