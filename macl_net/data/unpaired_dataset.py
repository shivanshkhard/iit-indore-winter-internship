import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class UnpairedDataset(Dataset):
    """
    Unpaired image dataset for MACL-Net / CycleGAN-style training.

    Structure expected:
        root/
            trainA/  (domain A images, e.g., horses)
            trainB/  (domain B images, e.g., zebras)
            testA/
            testB/
    """

    def __init__(
        self,
        root: str,
        phase: str = "train",
        img_size: int = 256,
        serial_batches: bool = False
    ):
        """
        Args:
            root (str): Path to dataset root (e.g., datasets/horse2zebra)
            phase (str): 'train' or 'test'
            img_size (int): Final cropped image size
            serial_batches (bool): If True, pairs A[i] with B[i]
                                   If False (default), random B sampling
        """
        super().__init__()

        self.phase = phase
        self.serial_batches = serial_batches

        self.dir_A = os.path.join(root, phase + "A")
        self.dir_B = os.path.join(root, phase + "B")

        if not os.path.isdir(self.dir_A) or not os.path.isdir(self.dir_B):
            raise FileNotFoundError(
                f"Dataset folders not found: {self.dir_A}, {self.dir_B}"
            )

        self.A_paths = sorted([
            os.path.join(self.dir_A, f)
            for f in os.listdir(self.dir_A)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        self.B_paths = sorted([
            os.path.join(self.dir_B, f)
            for f in os.listdir(self.dir_B)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.A_paths) == 0 or len(self.B_paths) == 0:
            raise RuntimeError("No images found in dataset folders.")

        # --------------------------------------------------
        # Transform pipeline (CycleGAN / MACL standard)
        # --------------------------------------------------
        if phase == "train":
            self.transform = T.Compose([
                T.Resize((286, 286), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)
                )
            ])

    def __len__(self):
        """
        Length defined as max of both domains
        (standard practice for unpaired training)
        """
        return max(len(self.A_paths), len(self.B_paths))

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Robust image loader (RGB enforced)
        """
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def __getitem__(self, index: int):
        """
        Returns:
            dict:
                A (Tensor): image from domain A
                B (Tensor): image from domain B
                A_path (str)
                B_path (str)
        """
        A_path = self.A_paths[index % len(self.A_paths)]

        if self.serial_batches:
            B_index = index % len(self.B_paths)
        else:
            B_index = random.randint(0, len(self.B_paths) - 1)

        B_path = self.B_paths[B_index]

        A_img = self._load_image(A_path)
        B_img = self._load_image(B_path)

        return {
            "A": A_img,
            "B": B_img,
            "A_path": A_path,
            "B_path": B_path
        }
