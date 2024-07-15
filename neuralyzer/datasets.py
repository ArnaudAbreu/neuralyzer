import numpy as np
import torch
from skimage.color import rgb2gray
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset
from typing import Callable, Iterator, Optional, Tuple
from torchvision.transforms import Compose


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


class PatchToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    
    def __init__(self, device: Optional[torch.device] = None, is_gray: bool = False):
        """
        Initialize the ToTensor transform.
        
        Args:
            device (torch.device): The device to store the tensor.
            is_gray (bool): Whether the input is grayscale
        """
        self.device = device
        self.is_gray = is_gray
    
    def __call__(self, sample: np.ndarray) -> torch.Tensor:
        """
        Call method of the ToTensor transform.
        
        Args:
            sample (np.ndarray): The input sample.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        result = torch.tensor(sample, dtype=torch.float32, device=self.device)
        if self.is_gray:
            result = result.unsqueeze(0)
        else:
            result = result.permute(2, 0, 1)
        return result


class PatchDataset_(Dataset):
    """
    A dataset that generates patches from an image.
    """
    def __init__(
        self,
        img: np.ndarray,
        patch_size: int,
        stride: int,
        transform: Optional[Callable] = None,
        shuffle: bool = True
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
        if shuffle:
            np.random.shuffle(self.grid)

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        y, x = self.grid[idx]
        patch = self.img[
            y:y + self.patch_size,
            x:x + self.patch_size
        ]
        if self.transform:
            patch = self.transform(patch)
        return {"image": patch, "position": (y, x)}


class GrayPatchDataset(object):
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
            transform (PatchToTensor): The transform to apply to the patches.
        """
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
    
    def load(self, img_path: str) -> DataLoader:
        """
        Load the image and create the patch dataset.
        
        Args:
            img_path (str): The path to the image.
        
        Returns:
            DataLoader: The patch dataset.
        """
        img = load_gray_image(img_path)
        dataset = PatchDataset_(
            img=img,
            patch_size=self.patch_size,
            stride=self.stride,
            transform=PatchToTensor(is_gray=True, device=self.device),
            shuffle=self.shuffle
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )
        return loader
    
    