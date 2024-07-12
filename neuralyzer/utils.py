import torch


class Patchifier(torch.nn.Module):
    """
    Patchifier: A module that converts an image into patches.
    """
    def __init__(self, patch_size: int, stride: int):
        """
        Initialize the Patchifier module.
        
        Args:
            patch_size (int): The size of the patch.
            stride (int): The stride of the patch.
        """
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Patchifier module.
        
        Args:
            x (torch.Tensor): The input image tensor.
            
        Returns:
            torch.Tensor: The patches of the image.
        """
        # x: (Batch, Channel, Height, Width)
        batch_size, channel, _, _ = x.shape
        x = self.unfold(x)
        # x: (Batch, Channel * PatchSize * PatchSize, NumPatches)
        patches = x.view(batch_size, channel, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        # a: (Batch, NumPatches, Channel, PatchSize, PatchSize)
        return patches