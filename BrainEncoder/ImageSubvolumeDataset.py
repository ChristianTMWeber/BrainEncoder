#import torch
from torch.utils.data import Dataset

from tifffile import imread
import os

import numpy as np

class ImageSubvolumeDataset(Dataset):
    def __init__(self, imageFilePath : "str", subvolumeSize = 50,  
        annotations_file = None, transform=None, target_transform=None):
        
        if annotations_file is not None: self.img_labels = pd.read_csv(annotations_file)
        else: self.img_labels= None
        
        self.imageFilePath = os.path.abspath(imageFilePath)
        self.image = imread(self.imageFilePath).astype(np.float16)

        self.subvolumeSize = subvolumeSize # assume for now same size in each direction

        nSegmentsPerAxis = [ len(self.segmentBoundaries(imageLength, self.subvolumeSize)) for imageLength in np.shape(self.image)  ]

        self.nSubvolumes = np.prod( nSegmentsPerAxis )


    
        self.transform = transform
        self.target_transform = target_transform


        self.subvolumeSlices = self.defineSubvolumes()

    def __len__(self):
        return self.nSubvolumes

    def __getitem__(self, idx):

        #sliceList = self.subvolumeSlices[idx]

        if not isinstance(idx,slice): idx = slice(idx,idx+1)
        
        subvolumeImages = [ self.image[imgSlice] for imgSlice in self.subvolumeSlices[idx] ]
        
        subvolumeImages = [np.average(image,2) for image in subvolumeImages] # collapse subvolume image to a 2d one

        #subvolumeImages = self.image[self.subvolumeSlices[idx]]

        #subvolumeImages = []
        #for subvolumeSlice in self.defineSubvolumes():
        #    subvolume = self.image[subvolumeSlice]
        #    subvolumeImages.append(subvolume)



        #subvolumeImage = self.image(self.defineSubvolumes())
        label = [0]*len(subvolumeImages)

        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = read_image(img_path)
        #label = self.img_labels.iloc[idx, 1]
        #if self.transform:
        #    image = self.transform(image)
        #if self.target_transform:
        #    label = self.target_transform(label)
        if len(subvolumeImages)==1:  return subvolumeImages[0], label[0]
        return subvolumeImages, label
    
    def segmentBoundaries(self, imageLength, segmentSize):
        return [x for x in range(0,imageLength-segmentSize+1,segmentSize)]
    

    def defineSubvolumes(self): #presume for now that we have a 3d image
        imageDimensions = np.shape(self.image)

        sliceList = []

        for xBoundary in self.segmentBoundaries(imageDimensions[0], self.subvolumeSize):
            for yBoundary in self.segmentBoundaries(imageDimensions[1], self.subvolumeSize):
                for zBoundary in self.segmentBoundaries(imageDimensions[2], self.subvolumeSize):

                    sliceTuple = tuple( [slice(xBoundary,xBoundary+self.subvolumeSize),
                                         slice(yBoundary,yBoundary+self.subvolumeSize),
                                         slice(zBoundary,zBoundary+self.subvolumeSize)])
                    
                    sliceList.append(sliceTuple)

        return sliceList
    

if __name__ == "__main__":


    imageDataset = ImageSubvolumeDataset("../NeuNBrainSegment_compressed.tiff")

    aSubvolume, subvolumeLabel = imageDataset[0]

    data = imageDataset[0:3]

    print("All Done!")

