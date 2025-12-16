"""
TinyImageNet Dataset Loader

Tiny ImageNet consists of:
- 200 classes (subset of ImageNet)
- 500 training images per class (100,000 total)
- 50 validation images per class (10,000 total)
- Images are 64x64 RGB

Dataset structure:
tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   │   └── images/
│   │       ├── n01443537_0.JPEG
│   │       └── ...
│   └── ...
├── val/
│   ├── images/
│   │   ├── val_0.JPEG
│   │   └── ...
│   └── val_annotations.txt
└── wnids.txt

Download: http://cs231n.stanford.edu/tiny-imagenet-200.zip
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image


class TinyImageNet(Dataset):
    """
    TinyImageNet dataset for self-supervised pretraining.

    For MAE pretraining, we don't need labels - only images.
    This loader supports both train and validation splits.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        download: bool = False
    ):
        """
        Args:
            root: Root directory of TinyImageNet dataset
            split: 'train' or 'val'
            transform: Torchvision transforms to apply
            download: If True, downloads the dataset (requires manual download)
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Check if dataset exists
        if not self.root.exists():
            if download:
                raise NotImplementedError(
                    "Automatic download not implemented. Please download manually from:\n"
                    "http://cs231n.stanford.edu/tiny-imagenet-200.zip\n"
                    f"Extract to: {self.root}"
                )
            else:
                raise FileNotFoundError(
                    f"TinyImageNet not found at {self.root}. "
                    "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
                )

        # Load image paths and labels
        self.samples = self._load_samples()

        print(f"Loaded TinyImageNet {split} split: {len(self.samples)} images")

    def _load_samples(self) -> list:
        """Load image paths and corresponding class labels."""
        samples = []

        if self.split == 'train':
            # Training set: organized by class folders
            train_dir = self.root / 'train'

            for class_dir in sorted(train_dir.iterdir()):
                if not class_dir.is_dir():
                    continue

                images_dir = class_dir / 'images'
                if not images_dir.exists():
                    continue

                for img_path in sorted(images_dir.glob('*.JPEG')):
                    samples.append((str(img_path), class_dir.name))

        elif self.split == 'val':
            # Validation set: all images in one folder with annotations file
            val_dir = self.root / 'val'
            annotations_file = val_dir / 'val_annotations.txt'

            # Parse annotations: img_name \t class_id \t x y w h
            img_to_class = {}
            with open(annotations_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    class_id = parts[1]
                    img_to_class[img_name] = class_id

            # Load image paths
            images_dir = val_dir / 'images'
            for img_name, class_id in img_to_class.items():
                img_path = images_dir / img_name
                if img_path.exists():
                    samples.append((str(img_path), class_id))

        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Index

        Returns:
            (image, label) tuple
            - For MAE pretraining, label is not used but included for compatibility
        """
        img_path, class_id = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # For pretraining, we don't need labels, but include dummy label for compatibility
        return image, 0


def get_tinyimagenet_stats():
    """
    Get dataset statistics for normalization.

    TinyImageNet uses ImageNet normalization:
    Mean: [0.485, 0.456, 0.406]
    Std: [0.229, 0.224, 0.225]
    """
    return {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }


def download_instructions():
    """Print instructions for downloading TinyImageNet."""
    print("="*70)
    print("TinyImageNet Dataset Download Instructions")
    print("="*70)
    print("\n1. Download the dataset:")
    print("   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    print("\n2. Extract the dataset:")
    print("   unzip tiny-imagenet-200.zip -d ./data/")
    print("\n3. Verify structure:")
    print("   data/tiny-imagenet-200/")
    print("   ├── train/")
    print("   ├── val/")
    print("   └── wnids.txt")
    print("\n4. The dataset will be ready to use!")
    print("="*70)


if __name__ == '__main__':
    """Test TinyImageNet loader."""
    import torchvision.transforms as transforms

    # Print download instructions
    download_instructions()

    # Try to load dataset
    root = './data/tiny-imagenet-200'

    if Path(root).exists():
        print(f"\nDataset found at {root}")

        # Create simple transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Test train set
        train_dataset = TinyImageNet(root=root, split='train', transform=transform)
        print(f"Train set size: {len(train_dataset)}")

        # Test val set
        val_dataset = TinyImageNet(root=root, split='val', transform=transform)
        print(f"Val set size: {len(val_dataset)}")

        # Test loading a sample
        img, label = train_dataset[0]
        print(f"Sample image shape: {img.shape}")
        print(f"Sample label: {label}")

        print("\n✓ TinyImageNet loader working correctly!")
    else:
        print(f"\nDataset not found at {root}")
        print("Please download using the instructions above.")
