import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

class Caravana(Dataset):
    def __init__(self,image_dir,mask_dir,transforms = None, resize_shape = (200,200)):
        super(Caravana,self).__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms= transforms
        self.resize_shape = resize_shape
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")) # as os.listdir returns in an arbitary order
        
        img = Image.open(img_path).convert("RGB").resize(self.resize_shape)
        mask = Image.open(mask_path).convert("L").resize(self.resize_shape)

        img = np.array(img)
        mask = np.array(mask,dtype=np.float32)
        mask[mask == 255.0] = 0.0

        if self.transforms is not None:
            augmentations = self.transforms(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]
        
        return img,mask

## Example Usage
if __name__ == "__main__":
    train_transform = A.Compose(
    [
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        # this will only divide by 255 (since mean = 0 and std = 1)
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
    dataset = Caravana("../data/imgs","../data/masks",resize_shape=(200,200),transforms=train_transform)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(dataset[0][0].permute(1, 2, 0).numpy())  # Convert tensor to numpy
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(dataset[0][1].numpy(), cmap='gray')  # Convert tensor to numpy and set cmap for grayscale
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.savefig("sample.png")
    plt.close()

