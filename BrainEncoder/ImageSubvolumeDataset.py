import torch
from torch.utils.data import Dataset
from torchvision import transforms # to manipulate input data


from tifffile import imread
import os

import numpy as np

class ImageSubvolumeDataset(Dataset):
    def __init__(self, imageFilePath : "str", subvolumeSize = 32,  
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


        if not isinstance(idx,slice): idx = slice(idx,idx+1)
        
        subvolumeImages = [ self.image[imgSlice] for imgSlice in self.subvolumeSlices[idx] ]
        

        #subvolumeImage = self.image(self.defineSubvolumes())
        label = [0]*len(subvolumeImages) # all labels are 0 for now


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
    
    # use 'collate_array' as 'collate_fn' argument 
    # when using the 'ImageSubvolumeDataset' as input to the 
    # torch.utils.data.DataLoader method
    #
    # i.e. 
    #   dataset = ImageSubvolumeDataset("NeuNBrainSegment_compressed.tiff", subvolumeSize = subvolumeSize)
    #
    #   torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"] , collate_fn=collate_array)
    #
    def collate_array(self,batch):

        def processArray(myArray):

            process = transforms.Compose([
                transforms.ToTensor(), 
                #transforms.Pad(2)
                ])

            return process(myArray)

        tensorList = [processArray(data[0]) for data in batch]


        imageBatchTensor = torch.stack(tensorList)  # Stacks along new axis, preserving 3D shape
        imageBatchTensor = imageBatchTensor.unsqueeze(1)  # Add channel dimension if needed

        #imageBatchTensor = torch.concat(tensorList).unsqueeze(1) # should have size [batchSize,1,imageXDim, imageYDim]

        # labels, note that we should convert the labels to LongTensor
        labelTensor = torch.LongTensor([data[1] for data in batch])

        return imageBatchTensor, labelTensor


if __name__ == "__main__":


    imageDataset = ImageSubvolumeDataset("../NeuNBrainSegment_compressed.tiff")

    aSubvolume, subvolumeLabel = imageDataset[0]

    data = imageDataset[0:3]

    print("All Done!")

