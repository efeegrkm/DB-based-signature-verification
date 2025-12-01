"""
siamese_dataset.py
====================

This module defines a PyTorch Dataset for building pairs of signature images
used to train a Siamese network. The dataset expects a directory structure
similar to the one used for the triplet‐loss implementation:

```
sign_data/
    split/
        train/
            001/
                image1.png
                image2.png
                ...
            001_forg/
                forg1.png
                forg2.png
                ...
            002/
                ...
            002_forg/
                ...
        val/
            ...
        test/
            ...
```

For each real user directory (e.g. ``001``) the dataset will generate
``positive`` pairs by sampling two distinct genuine images from the same
directory. ``Negative`` pairs can be generated in two ways:

* **Real–Forge negatives:** A genuine signature from a user is paired with one
  of that user’s forgery images. These negatives are useful because they
  represent the most challenging impostor attempts.
* **Cross‐user negatives:** Two images belonging to different users (where
  forgeries are considered separate from genuine classes) are paired together.

The dataset returns a tuple ``(img1, img2, label)``. ``label`` follows the
convention used by the contrastive loss: ``0`` for a genuine pair (similar) and
``1`` for a negative pair (dissimilar).

You can control basic augmentation by passing a torchvision transform. For
training it is recommended to include random rotation, translation and color
shifts, similar to the augmentation used in the triplet implementation.

Example::

    from siamese_dataset import SignatureSiameseDataset
    import torchvision.transforms as transforms

    train_transforms = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = SignatureSiameseDataset('sign_data/split/train', transform=train_transforms)
    img1, img2, label = dataset[0]
"""

import os
import random
from typing import Dict, List, Tuple, Optional

from PIL import Image
from torch.utils.data import Dataset


class SignatureSiameseDataset(Dataset):
    """Dataset generating signature pairs for Siamese network training.

    Each element is a tuple consisting of two preprocessed images and a label.

    Args:
        root_dir: Path to the directory containing user folders. The directory
            should contain subdirectories for each user (e.g. ``001``) and an
            optional ``*_forg`` directory for that user's forgeries.
        transform: Optional torchvision transform applied to both images.
        pos_fraction: Fraction of samples that should be positive pairs. Must
            be in the interval (0, 1). Defaults to 0.5.
    """

    def __init__(self,
                 root_dir: str,
                 transform=None,
                 pos_fraction: float = 0.5) -> None:
        self.root_dir: str = root_dir
        self.transform = transform
        if not 0.0 < pos_fraction < 1.0:
            raise ValueError("pos_fraction must be between 0 and 1, exclusive.")
        self.pos_fraction: float = pos_fraction

        # Build mappings from user names to lists of real and forgery images.
        self.user_real_images: Dict[str, List[str]] = {}
        self.user_forg_images: Dict[str, List[str]] = {}

        # Populate dictionaries by scanning the directory.
        # Only consider image files with common extensions.
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        for entry in os.listdir(self.root_dir):
            entry_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry.endswith('_forg'):
                user_name = entry[:-5]
                # List forgery images
                imgs = [os.path.join(entry_path, f)
                        for f in os.listdir(entry_path)
                        if f.lower().endswith(valid_exts)]
                if imgs:
                    self.user_forg_images[user_name] = imgs
            else:
                user_name = entry
                # List real images
                imgs = [os.path.join(entry_path, f)
                        for f in os.listdir(entry_path)
                        if f.lower().endswith(valid_exts)]
                if imgs:
                    self.user_real_images[user_name] = imgs

        # Construct a list of all classes: tuples ("real", user) or ("forg", user)
        # Cross‐user negatives will be drawn from this list.
        self.all_classes: List[Tuple[str, str]] = []
        for user in self.user_real_images:
            self.all_classes.append(('real', user))
            if user in self.user_forg_images:
                # treat forgery class separately to allow cross negatives
                self.all_classes.append(('forg', user))

        # Precompute a list of users with at least two real images for positives.
        self.users_with_multiple_real: List[str] = [u for u, imgs in self.user_real_images.items() if len(imgs) >= 2]
        if not self.users_with_multiple_real:
            raise RuntimeError("No user has at least two real images in the provided dataset.")

        # Determine a nominal dataset length. We choose to make one sample per genuine
        # signature on average, though training loops can iterate indefinitely.
        self._length: int = sum(len(imgs) for imgs in self.user_real_images.values())

    def __len__(self) -> int:
        return self._length

    def _get_positive_pair(self) -> Tuple[str, str]:
        """Randomly selects and returns two distinct genuine images from the same user.

        Returns a pair of file paths for images and the corresponding label (0).
        """
        # Choose a user who has at least two real images
        user = random.choice(self.users_with_multiple_real)
        imgs = self.user_real_images[user]
        img1_path, img2_path = random.sample(imgs, 2)
        return img1_path, img2_path

    def _get_negative_pair(self) -> Tuple[str, str]:
        """Randomly selects and returns two images representing a negative pair.

        The function will either produce a real–forgery pair for the same
        user (if forgery data exists) or a cross‐user pair drawn from the
        combined list of classes (real and forgery classes). Forgery–forgery
        pairs are treated as cross‐user negatives by considering each forgery
        directory as its own class.
        """
        # With 50 % chance, produce a real–forgery negative if possible
        use_forgery = False
        # Only users with forgery images are eligible for real–forgery negatives
        users_with_forg = [u for u in self.user_forg_images.keys() if self.user_forg_images[u] and self.user_real_images.get(u)]
        if users_with_forg and random.random() < 0.5:
            use_forgery = True

        if use_forgery:
            user = random.choice(users_with_forg)
            real_img = random.choice(self.user_real_images[user])
            forg_img = random.choice(self.user_forg_images[user])
            return real_img, forg_img

        # Otherwise, sample two distinct classes for a cross‐user negative
        # Each class is a tuple ('real'/'forg', user_name)
        class1, class2 = random.sample(self.all_classes, 2)
        img1 = None
        img2 = None
        # Select an image for class1
        if class1[0] == 'real':
            img1 = random.choice(self.user_real_images[class1[1]])
        else:
            # forgery class
            img1 = random.choice(self.user_forg_images[class1[1]])
        # Select an image for class2
        if class2[0] == 'real':
            img2 = random.choice(self.user_real_images[class2[1]])
        else:
            img2 = random.choice(self.user_forg_images[class2[1]])
        return img1, img2

    def __getitem__(self, index: int):
        # Determine whether to sample a positive or negative pair
        is_positive = random.random() < self.pos_fraction
        if is_positive:
            img1_path, img2_path = self._get_positive_pair()
            label = 0  # similar for contrastive loss
        else:
            img1_path, img2_path = self._get_negative_pair()
            label = 1  # dissimilar for contrastive loss

        # Load and process images
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label