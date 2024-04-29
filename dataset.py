import torch
import utils
from torch.utils.data import Dataset
import os
import cv2 as cv


class NoisyDataset(Dataset):
    """
    A subclass of the Dataset class of pytorch. 
        Input : dirPath : The folder path where the images in the dataset are stored

                stdRange : A tuple representing the range of standard deviation of gaussian noise added to each image read.

                batchSize : The batch size of patches 

                pathDimension : The desired dimension of each patch

        This class uses two functions implemented in the utils.py file
    """

    def __init__(self, dirPath : str, stdRange : tuple, batchSize : int, patchDimension : int) -> None:

        super().__init__()
        
        self.dirPath = dirPath
        self.batchSize = batchSize
        self.stdRange = stdRange
        self.patchDim = patchDimension

        self.cleanImages = []  # the list of torcg.tensor scaled in the range 0,1 and representing the images in the dataset

        self.noisyImages = [] #  the list of torch.tensors scaled in the range 0,1 and with guassian noise applied

        self.groundTruth = []  # a list of patches. each patch is a scaled tensor of shape (3, 50 ,50)

        self.input = []    # a list of noisy patches.  each patch is a scaled tensor of shape (3, 50 ,50)

        self.readImages()
        self.extractPatch()
    

    def readImages(self):

        listOfImagesPath = os.listdir(self.dirPath)  # a list of images for the specified director


        for pathToImage in listOfImagesPath:

            npImage = cv.imread(self.dirPath + "/" + pathToImage)

            torchImage = torch.from_numpy(npImage).float()

            torchImage = torchImage.permute(2,0,1) / 255

            self.cleanImages.append(torchImage)

            noisyImage = utils.applyRandomNoise(torchImage, stdRange=self.stdRange)

            self.noisyImages.append(noisyImage)

    
    def extractPatch(self):

        for image in self.cleanImages:

            listOfPatch = utils.extractPatchesFromTensor(self.patchDim, image)
            
            self.groundTruth += listOfPatch

        for noisyImage in self.noisyImages:

            listOfPatch = utils.extractPatchesFromTensor(self.patchDim, noisyImage)
            
            self.input += listOfPatch 
        
    def __len__(self):
        
        return len(self.input)
    

    def __getitem__(self, index) :

        # a list of len (bathSize). each element of the list is a tensor of shape (3, patchdim, patchdim)

        input = self.input[index : index + self.batchSize]  

        groundT = self.groundTruth[index : index + self.batchSize]


        batchedInput = torch.stack(input, dim=0)

        batchedGroundTruth = torch.stack(groundT, dim=0)

        return batchedInput, batchedGroundTruth



    
