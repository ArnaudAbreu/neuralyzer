from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.util import img_as_float
from torch.utils.data import DataLoader, Dataset


def load_gray_image(path: str) -> np.ndarray:
    """
    Load an image from a file and convert it to grayscale.
    
    Args:
        path (str): The path to the image file.
    
    Returns:
        np.ndarray: The grayscale image.
    """
    img = imread(path)
    return rgb2gray(img)


def load_rgb_image(path: str) -> np.ndarray:
    """
    Load an image from a file and convert it to RGB.
    
    Args:
        path (str): The path to the image file.
    
    Returns:
        np.ndarray: The RGB image.
    """
    img = imread(path)
    return img_as_float(img)


def load_image(path: str, gray: bool = False) -> np.ndarray:
    """
    Load an image from a file.
    
    Args:
        path (str): The path to the image file.
        gray (bool): Whether to convert the image to grayscale.
    
    Returns:
        np.ndarray: The image.
    """
    if gray:
        return load_gray_image(path)
    return load_rgb_image(path)


def sample_line(size: int, stride: int) -> np.ndarray:
    """
    Sample discrete positions on a discrete finite line.
    
    Args:
        size: The size of the line.
        stride: The stride between samples.
    """
    # Use linespace to get the positions of the samples.
    # We do not center the samples
    return np.linspace(0, size, size // stride, dtype=np.int32, endpoint=False)


def sample_grid(height: int, width: int, stride: int) -> Iterator[Tuple[int, int]]:
    """
    Sample discrete positions on a discrete finite grid.
    
    Args:
        height: The height of the grid.
        width: The width of the grid.
        stride: The stride between samples.
    """
    # Get the positions of the samples on the x and y axis.
    x = sample_line(width, stride)
    y = sample_line(height, stride)
    # double loop to yield the cartesian product of the samples
    # oneliner
    return [(i, j) for i in y for j in x]


class PatchToFlatTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the GrayPatchToFlatTensor transform.
        
        Args:
            device (torch.device): The device to store the tensor.
            is_gray (bool): Whether the input is grayscale
        """
        self.device = device
    
    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        """
        Transform a gray patch to a flat torch tensor.
        
        Args:
            sample (np.ndarray): The input sample.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        flat = sample.flatten()
        result = torch.tensor(flat, dtype=torch.float32, device=self.device)
        return result


class PatchDataset(Dataset):
    """
    A dataset that generates patches from an image.
    """
    def __init__(
        self,
        img: np.ndarray,
        patch_size: int,
        stride: int,
        shuffle: bool = True,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the PatchDataset.
        
        Args:
            img (np.ndarray): The input image.
            patch_size (int): The size of the patch.
            stride (int): The stride of the patch.
            transform (Optional[Callable]): The transform to apply to the patches.
        
        Returns:
            torch.Tensor: The output
        """
        self.transform = transform
        self.img = img
        self.patch_size = patch_size
        self.stride = stride
        self.grid = sample_grid(
            img.shape[0] - patch_size, 
            img.shape[1] - patch_size,
            stride
        )
        self.shuffle = shuffle
        self.shuffle_indices = np.arange(len(self.grid))
        np.random.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        real_idx = idx
        if self.shuffle:
            real_idx = self.shuffle_indices[idx]
        y, x = self.grid[real_idx]
        patch = self.img[
            y:y + self.patch_size,
            x:x + self.patch_size
        ]
        if self.transform:
            patch = self.transform(patch)
        return {"image": patch, "position": (y, x)}
    
    def loc(self, x: int, y: int) -> np.ndarray:
        """
        Get the patch at a specific location.
        
        Args:
            x (int): The x coordinate of the patch.
            y (int): The y coordinate of the patch.
        
        Returns:
            np.ndarray: The patch.
        """
        patch = self.img[
            y:y + self.patch_size,
            x:x + self.patch_size
        ]
        return patch


class FlatPatchDataset(object):
    """
    A patch data loader.
    """
    def __init__(
        self,
        patch_size: int,
        stride: int,
        batch_size: int,
        shuffle: bool = True,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initialize the PatchDataset.
        
        Args:
            patch_size (int): The size of the patch.
            stride (int): The stride of the patch.
            batch_size (int): The batch size.
            shuffle (bool): Whether to shuffle the patches.
            device (torch.device): The device to store the tensor.
        """
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    
    def load(
        self,
        img_path: str,
        gray: bool = False) -> DataLoader:
        """
        Load the image and create the patch dataset.
        
        Args:
            img_path (str): The path to the image.
            gray (bool): Whether the image is grayscale.
        
        Returns:
            DataLoader: The patch dataset.
        """
        img = load_image(img_path, gray=gray)
        dataset = PatchDataset(
            img=img,
            patch_size=self.patch_size,
            stride=self.stride,
            transform=PatchToFlatTensor(device=self.device),
            shuffle=self.shuffle
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            
        )
        return loader
    
    