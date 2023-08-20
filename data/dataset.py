import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


TRAIN_IMG_DIR = "Full_no_black/train_images/"
TRAIN_MASK_DIR = "Full_no_black/train_masks/"
VAL_IMG_DIR = "Full_no_black/val_images/"
VAL_MASK_DIR = "Full_no_black/val_masks/"


class Brain_MRI_Segmentation_Dataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform=None):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.transform = transform
		self.images = os.listdir(image_dir)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.images[index].replace(".tif", "_mask.tif"))
		image = np.array(Image.open(img_path).convert("RGB"))
		mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
		mask[mask == 255.0] = 1.0

		if self.transform is not None:
			augmentations = self.transform(image=image, mask=mask)
			image = augmentations["image"]
			mask = augmentations["mask"]

		return image, mask



def get_loaders(
	batch_size,
	num_workers=4,
	pin_memory=True,
):
	
	train_transform = transforms.Compose(
				[
					transforms.RandomRotation(degrees=35),
					transforms.RandomHorizontalFlip(p=0.5),
					transforms.RandomVerticalFlip(p=0.1),
                    transforms.Scale(),
					transforms.ToTensor(),
                    transforms.Normalize(
						mean=[0.0, 0.0, 0.0],
						std=[1.0, 1.0, 1.0],
					),
				],
			)
	
	val_transform = transforms.Compose(
				[
                    transforms.ToTensor(),
					transforms.Normalize(
						mean=[0.0, 0.0, 0.0],
						std=[1.0, 1.0, 1.0],
					),
				],
			)
	
	train_ds = Brain_MRI_Segmentation_Dataset(
		image_dir=TRAIN_IMG_DIR,
		mask_dir=TRAIN_MASK_DIR,
		transform=train_transform,
	)

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=True,
	)

	val_ds = Brain_MRI_Segmentation_Dataset(
		image_dir=VAL_IMG_DIR,
		mask_dir=VAL_MASK_DIR,
		transform=val_transform,
	)

	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		num_workers=num_workers,
		pin_memory=pin_memory,
		shuffle=False,
	)

	return train_loader, val_loader

