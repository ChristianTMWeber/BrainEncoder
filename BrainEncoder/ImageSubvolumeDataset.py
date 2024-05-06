import torch
from torch.utils.data import Dataset
from torchvision import transforms # to manipulate input data

import os

import numpy as np
import matplotlib.pyplot as plt

import time

# for monitoring memory usage
import tracemalloc

from tiffStackArray import tiffStackArray # use this to load the stack of tiff files

class ImageSubvolumeDataset(Dataset):
    def __init__(self, imageFilePath : "str", subvolumeSize = 32,  
        minimumFractionalFill : "float" = None , 
        regionOfRelevance : "tuple(slice)" = False ):
        # minimum fraction of non-zero pixels in a subvolume for it to be 
        #   included in the dataset
        # regionOfRelevance - if we don't want to consider the full information / all 
        #  voxels represented by the data under 'imageFilePath', we can supply
        #  a tuple of slice objects, e.g.
        #  regionOfRelevance = (slice(z0:z1), slice(y0:y1), slice(x0:x1))
        #  This selects the subregion between 
        #       z0 and z1 along z
        #       y0 and y1 along z
        #       x0 and x1 along z
        # Note the axis definitions that we observe. I.e.
        # dimension 0 - is along Z
        # dimension 1 - is along Y
        # dimension 2 - is along X
        # pass regionOfRelevance = slice(None) to copy the full tiff information into memory
        
        self.img_labels= None
        
        self.imageFilePath = os.path.abspath(imageFilePath)

        #self.imageNPArray = self.loadInputImages(self.imageFilePath)
        
        self.imageNPArray = tiffStackArray(self.imageFilePath)

        # commit part or all of the tiff information to memory
        if isinstance(regionOfRelevance,slice):
            startime = time.time()
            self.imageNPArray = self.imageNPArray[ regionOfRelevance ]
            print("Array copy time: %i s" % (time.time()-startime))

        self.subvolumeSize = subvolumeSize # assume for now same size in each direction
        self.minimumFractionalFill = minimumFractionalFill


        startime = time.time()
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB; Diff = {(peak - current) / 10**6}MB")
        print("Defining Subvolumes")
        self.subvolumeSlices = self.defineSubvolumes(self.subvolumeSize,self.minimumFractionalFill)
        print("Completed subvolume definition, elapsed time: %i s" % (time.time()-startime))
        current, peak = tracemalloc.get_traced_memory()

        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB; Diff = {(peak - current) / 10**6}MB")

        self.nSubvolumes = len(self.subvolumeSlices)
        self.potentialSubvolumes = np.prod(np.asarray(self.imageNPArray.shape)//32)

        self.minimumFractionalFill_Efficiency = self.nSubvolumes / self.potentialSubvolumes

        return None


    def __len__(self):
        return len(self.subvolumeSlices)

    def __getitem__(self, idx):


        if not isinstance(idx,slice): idx = slice(idx,idx+1)
        
        subvolumeImages = [ self.imageNPArray[imgSlice] for imgSlice in self.subvolumeSlices[idx] ]
        

        #subvolumeImage = self.imageNPArray(self.defineSubvolumes())
        label = [0]*len(subvolumeImages) # all labels are 0 for now


        if len(subvolumeImages)==1:  return subvolumeImages[0], label[0]
        return subvolumeImages, label


    def segmentBoundaries(self, imageLength, segmentSize):
        return [x for x in range(0,imageLength-segmentSize+1,segmentSize)]
    

    def defineSubvolumes(self,subvolumeSize,minimumFractionalFill): #presume for now that we have a 3d image
        
        def segmentIsTooEmpty(segment: "np.array", emptyThreshold = 0.001) -> "bool":

            fractionalFill = np.sum( segment > 0 ) / np.prod(np.shape(segment))

            return fractionalFill < emptyThreshold
        

        imageDimensions = np.shape(self.imageNPArray)

        sliceList = []
        
        count = 0

        for zBoundary in self.segmentBoundaries(imageDimensions[0], subvolumeSize):

            # we define here a zSlice thick enough for our subvolume size to speed up 
            # the segment definition in case we have minimumFractionalFill != None
            # this zSlice is loaded directly into memory, and thus can loop over 
            # all of the y and x boundaries quickly
            zSlice = self.imageNPArray[slice(zBoundary,zBoundary+subvolumeSize)]

            for yBoundary in self.segmentBoundaries(imageDimensions[1], subvolumeSize):
                for xBoundary in self.segmentBoundaries(imageDimensions[2], subvolumeSize):

                    sliceTuple = tuple( [slice(zBoundary,zBoundary+subvolumeSize),
                                         slice(yBoundary,yBoundary+subvolumeSize),
                                         slice(xBoundary,xBoundary+subvolumeSize)])
                    
                    sliceTupleYX = tuple([slice(None)] + list(sliceTuple[1:]))

                    if minimumFractionalFill is not None:
                        if segmentIsTooEmpty(zSlice[sliceTupleYX], emptyThreshold = minimumFractionalFill): continue
                    
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

    imageFilePath = "../NeuNBrainSegment_compressed.tiff"
    #imageFilePath = "/media/ssdshare1/general/computational_projects/brain_segmentation/DaeHee_NeuN_data/20190621_11_58_27_349_fullbrain_488LP30_561LP140_642LP100/Ex_2_Em_2_destriped_stitched_master"


    imageDataset = ImageSubvolumeDataset(imageFilePath, minimumFractionalFill= 1E-4 )
    #imageDataset = ImageSubvolumeDataset(imageFilePath, minimumFractionalFill= 1E-4 , regionOfRelevance=slice(None))

    print( "There are %i elements in the dataset" %len(imageDataset) )

    aSubvolume, subvolumeLabel = imageDataset[0]

    data = imageDataset[0:3]

    print("All Done!")

