import torch
from torch.utils.data import Dataset
from torchvision import transforms # to manipulate input data

import os

import numpy as np
import matplotlib.pyplot as plt

import time
import scipy.ndimage

# for monitoring memory usage
import tracemalloc

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tiffStackArray import tiffStackArray # use this to load the stack of tiff files

class ImageSubvolumeDataset(Dataset):
    def __init__(self, imageFilePath : "str", subvolumeSize = 32,  
        minimumFractionalFill : "float" = None , 
        regionOfRelevance : "tuple(slice)" = False,
        batchSize = 64, randomBatches = False ,
        nAugmentedSamples = 5):
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
        if isinstance(regionOfRelevance,slice) or all([isinstance(x,slice) for x in regionOfRelevance]):
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
        print("There are %i subvolumes" %self.nSubvolumes )
        self.potentialSubvolumes = np.prod(np.asarray(self.imageNPArray.shape)//32)

        self.minimumFractionalFill_Efficiency = self.nSubvolumes / self.potentialSubvolumes

        self.indexRange = range(0,max(np.shape(self.imageNPArray))) # use this to convert slices to intervals

        self.batchSize = batchSize
        self.randomBatches = randomBatches

        self.nAugmentedSamples = nAugmentedSamples

        self.nYieldBatches = (self.nSubvolumes-nAugmentedSamples)//self.batchSize

        self.randomIndices = np.random.permutation(self.nSubvolumes)

        if self.randomBatches: self.indexMapper = lambda x: self.randomIndices[x]
        else: self.indexMapper = lambda x: x

        return None
    

    

    def __len__(self):
        return len(self.subvolumeSlices)

    def __getitem__(self, idx):


        if not isinstance(idx,slice): idx = slice(idx,idx+1)
        

        [self.indexRange[imSlice[0]][0] for imSlice in [imgSliceTuple for imgSliceTuple in self.subvolumeSlices[idx]]]

        # each tuple in this list comproses three slice objects (sliceZ, sliceY, sliceX)
        # sliceZ selects the array elements of the given subvolume along Z
        # sliceY, sliceX do the same along Y and X
        list_of_ZYXsliceTuples = [imSlice for imSlice in  [imgSliceTuple for imgSliceTuple in self.subvolumeSlices[idx]]]

        subvolumeImages = [ self.imageNPArray[imgSliceTuple] for imgSliceTuple in list_of_ZYXsliceTuples ]

        subvolumeImagesNormalized = []
        # normalize the subvolume
        for image in subvolumeImages:

            image = image - np.min(image)
            image = image / (np.max(image) + 1E-5) * 100

            subvolumeImagesNormalized.append(image)
            
        subvolumeImages = subvolumeImagesNormalized


        subvolumeLabels = [] # each label is a tuple of z,y, and x coordinates
        for imgSliceTuple in list_of_ZYXsliceTuples: # imgSliceTuple is a tuple of slices: (sliceZ, sliceY, sliceX)
            subvolumeLabels.append( tuple( [self.indexRange[imSlice][0] for imSlice in imgSliceTuple ] ) )

        # if only one element was requested, we don't want to return a list of that single subvolume / label
        if len(subvolumeImages)==1:  return subvolumeImages[0], subvolumeLabels[0]
        return subvolumeImages, subvolumeLabels


    def segmentBoundaries(self, imageLength, segmentSize):
        return [x for x in range(0,imageLength-segmentSize+1,segmentSize)]
    

    def defineSubvolumes(self,subvolumeSize,minimumFractionalFill): #presume for now that we have a 3d image
        
        def segmentIsTooEmpty(segment: "np.array", emptyThreshold = 0.001) -> "bool":

            fractionalFill = np.sum( segment > 0 ) / np.prod(np.shape(segment))

            return fractionalFill < emptyThreshold
        

        imageDimensions = np.shape(self.imageNPArray)

        sliceList = []
        

        import subprocess

        for zBoundary in self.segmentBoundaries(imageDimensions[0], subvolumeSize):

            # we define here a zSlice thick enough for our subvolume size to speed up 
            # the segment definition in case we have minimumFractionalFill != None
            # this zSlice is loaded directly into memory, and thus can loop over 
            # all of the y and x boundaries quickly
            zSlice = self.imageNPArray[slice(zBoundary,zBoundary+subvolumeSize)]
            print("zBoundary = %i" %zBoundary)

            command = "lsof -u chweber 2>/dev/null | wc -l"
            result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE)
            print(result.stdout.strip())

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

            return process(myArray.astype(np.single))

        tensorList = [processArray(data[0]) for data in batch]


        imageBatchTensor = torch.stack(tensorList)  # Stacks along new axis, preserving 3D shape
        imageBatchTensor = imageBatchTensor.unsqueeze(1)  # Add channel dimension if needed

        #imageBatchTensor = torch.concat(tensorList).unsqueeze(1) # should have size [batchSize,1,imageXDim, imageYDim]

        # labels, note that we should convert the labels to LongTensor
        labelTensor = torch.LongTensor([data[1] for data in batch])

        return imageBatchTensor, labelTensor


    def transformImage(self, image, transformIndex):

        transformIndex = transformIndex % 10
        
        if transformIndex < 8:

            rotationAxes = np.random.choice(range(0,np.ndim(image)), 2, replace=False)
            rotationAngle = 90*np.random.randint(1,4)

            transformedImage = scipy.ndimage.rotate(image, angle=rotationAngle, axes=rotationAxes, reshape=True)


        else: 
            # flip one axis of the image
            permutation = np.random.permutation((slice(None,None,-1), slice(None),slice(None)))

            transformedImage = image[tuple(permutation)]
            
        return transformedImage


    # I want to yield subsets of the dataset, where each subset is a batch of subvolumes
    def __iter__(self):


        # lambda function that returns input data for a given index

        nNewSamplesPerBatch = self.batchSize - self.nAugmentedSamples


        for batchNr in range(0, self.nSubvolumes-nNewSamplesPerBatch, nNewSamplesPerBatch):


            sliceIndices = [self.indexMapper(x) for x in range(batchNr,batchNr+nNewSamplesPerBatch)]
            sliceList = [ self.subvolumeSlices[sliceIndex] for sliceIndex in sliceIndices]

            imageList = [self.imageNPArray[imgSlice] for imgSlice in sliceList]

            # augment the data
            # select x out of n without replacement
            augmentedIndices = np.random.choice(nNewSamplesPerBatch, self.nAugmentedSamples, replace=True)

            augmentedImageList = []
            augmentedSliceList = []

            for augIndex in augmentedIndices:
                #imgSlice = sliceList[augIndex]
                imgSlice = sliceList[0]

                image = self.imageNPArray[imgSlice]

                #random integer 
                augmentedImageList.append(self.transformImage(image, np.random.randint(0,255)))
                augmentedSliceList.append(imgSlice)
            #sliceList = [    self.subvolumeSlices ]

            imageList.extend(augmentedImageList)
            sliceList.extend(augmentedSliceList)

            subvolumeLabels = [] # each label is a tuple of z,y, and x coordinates
            for imgSliceTuple in sliceList: # imgSliceTuple is a tuple of slices: (sliceZ, sliceY, sliceX)
                subvolumeLabels.append( tuple( [self.indexRange[imSlice][0] for imSlice in imgSliceTuple ] ) )

            imageAndLabelList = [ (image,label) for image,label in zip(imageList,subvolumeLabels)]

            # turn the list of arrays and slices into tensors
            yield self.collate_array(imageAndLabelList)



if __name__ == "__main__":

    script_path = os.path.dirname(os.path.abspath(__file__))

    #imageFilePath = os.path.join(script_path,"../NeuNBrainSegment_compressed.tiff")
    imageFilePath = "/media/ssdshare1/general/computational_projects/brain_segmentation/DaeHee_NeuN_data/20190621_11_58_27_349_fullbrain_488LP30_561LP140_642LP100/Ex_2_Em_2_destriped_stitched_master"


    imageDataset = ImageSubvolumeDataset(imageFilePath, minimumFractionalFill= 1E-4,
                                        regionOfRelevance=(slice(1000,1064), slice(2000,2000+3000),slice(950,900+2700)),
                                        batchSize = 64, randomBatches = True ,
                                        nAugmentedSamples = 5 )
    #imageDataset = ImageSubvolumeDataset(imageFilePath, minimumFractionalFill= 1E-4 , regionOfRelevance=slice(None))

    print( "There are %i elements in the dataset" %len(imageDataset) )

    data = imageDataset[0:3]


    aSubvolume, subvolumeLabel = imageDataset[0]

    for batch in imageDataset:
        #print("Batch")
        #break
        pass


    #labelA   =  data[1][0]
    #labelB  =  data[1][1]

    print("All Done!")

