import torch
import random
from torch.nn.functional import pad

def extractPatchesFromTensor(patch_dim : int, tensor : torch.Tensor):

        """
        A function taking as input a tensor representing a image. Its going to extract and return a list of n patches of dimension patch_dim

        The padding strategy for both dimension is the following: if the created patch is composed of more padded values than real values, the patch is 
        not going to get created. 

        """
        listOfPatches  = []

        if tensor.shape[2] == 3:
            tensor = tensor.reshape(tensor.shape[2], tensor.shape[0], tensor.shape[1])
        
        shape = tensor.shape
        # we now have a shape for the tensor that is (3, H, W)

        horizontalIndex = shape[2] / patch_dim
        verticalIndex = shape[1] / patch_dim

        verticalPixelsToPad = patch_dim * (int(verticalIndex) + 1) - shape[1]

        horizontalPixelsToPad = patch_dim * (int(horizontalIndex) + 1) - shape[2]
        
        if patch_dim / 2 > verticalPixelsToPad:
            # we can pad ont he bottom of the tensor
            #pad = left, rigjt, top, bottom
            
            tensor = pad(tensor, pad=(0, 0, 0, verticalPixelsToPad), mode="reflect")
            verticalIndex += 1

        if patch_dim / 2 > horizontalPixelsToPad:
            # we can pad the left side of the tensor
            tensor = pad(tensor, pad=(0, horizontalPixelsToPad, 0, 0), mode="reflect" )
            horizontalIndex += 1


        for verticalSliceIndex in range(1, int(verticalIndex)):
            for horizontalSliceIndex in range(1, int(horizontalIndex)):
                
                patch = tensor[: , 
                            (verticalSliceIndex - 1) * patch_dim : verticalSliceIndex * patch_dim , 

                            (horizontalSliceIndex - 1) * patch_dim : horizontalSliceIndex * patch_dim
                            ]
                
                listOfPatches.append(patch)

        
        return listOfPatches



def applyRandomNoise(image, stdRange=(0, 55)):
    std_min = stdRange[0] / 255
    std_max = stdRange[1] / 255

    # Generate a random standard deviation
    std = torch.rand(1) * (std_max - std_min) + std_min

    # Sample Gaussian noise with the random standard deviation
    noise = torch.randn_like(image) * std 
    noisy_image = image + noise
    return torch.clamp(noisy_image, 0, 1)



import matplotlib.pyplot as plt

def plot_image(tensor):
    # Ensure tensor is in CPU and convert to numpy array
    image = tensor.detach().cpu().numpy()

    # If the tensor has a channel dimension, remove it
    if image.shape[0] == 1:
        image = image.squeeze(0)
    elif image.shape[0] == 3:
        # Convert from (C, H, W) to (H, W, C) format
        image = image.transpose(1, 2, 0)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

