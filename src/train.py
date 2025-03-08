from architecture.unet import UNet
from data_processing import Caravana
from torch.utils.data import DataLoader
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
import torch.nn as nn
from loguru import logger
import torch
from tqdm import tqdm

def train(model,train_loader,device,optimizer,num_epochs=1):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for images,masks in train_loader:
            images , masks = images.to(device),masks.to(device)
            optimizer.zero_grad()
            # logger.debug(type(images))
            # logger.debug(f"{images.shape},{masks.shape}")
            out = model(images)
            # logger.debug(out.shape[2])
            masks = masks.unsqueeze(0)  # Ensures shape is (N, C, H, W)
            # logger.debug(masks.shape)
            resized_tensor = F.interpolate(masks, size=(out.shape[2], out.shape[2]), mode="bilinear", align_corners=False)
            # logger.debug(resized_tensor.shape)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(resized_tensor,out)
            # logger.debug(loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()        
        logger.info(f"Epoch : {epoch}, Loss : {total_loss/len(train_loader)}")

#Example Usage

image_dir = "data/imgs"
mask_dir = "data/masks"
train_transform = A.Compose(
    [
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
train_dataset = Caravana(image_dir=image_dir,mask_dir=mask_dir,transforms=train_transform,resize_shape=(200,200))
train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True)
device = "cuda"

model = UNet(device=device,n_class=1).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-2)
train(model=model,
      train_loader=train_loader,
      device=device,
      optimizer=optimizer,
      num_epochs=1)
