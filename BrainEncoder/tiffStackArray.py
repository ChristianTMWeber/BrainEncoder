import os
import zarr
import numpy as np
from tifffile import imread

import re

#import psutil # check available memory with: psutil.virtual_memory()
# for monitoring memory usage


class tiffStackArray():

    def __init__(self, imagePath : str , fileType : "str" = "\.tif{1,2}" ):
        # imagePath should be path to a tifffile or path to a folder with tifffiles
        # fileType is a regular expression string that determines the file type we are looking for
        # change it in case we don't want to open a tifffile

        self.imagePath = imagePath
        self.fileType = fileType

        # get a list of the relevant files
        self.tiffFileTuple = self.getTiffFileTuple(imagePath,fileType)

        # tuple of 'zarr.core.Array' that we can address like an array from disk
        # so we can access many very large arrays, without running out of memory
        self.zarrFileTuple = self.openTiffsAsZarrs(self.tiffFileTuple)

        self.dtype = self.zarrFileTuple[0].dtype

        shape = []
        if len(self.zarrFileTuple) > 1: shape.append(len(self.zarrFileTuple))
        shape.extend(np.shape(self.zarrFileTuple[0]))
        self.shape = tuple(shape)

        return None
    
    def __len__(self): return np.prod(self.shape)


    def __getitem__(self, idx : int | slice | tuple) -> np.array:

        # when setting up the access to the stack of tiff files it bears to keep in mind 
        # how numpy orders multidimensional arrays
        # np.zeros([3,3]) is an array of arrays  (array^2)
        #   so np.zeros([3,3])[0] yields an array
        #
        # np.zeros([3,3,3]) is an array of arrays of arrays    (array^3)
        #   so np.zeros([3,3,3])[0] yields an array of arrays  (array^2)
        #   and np.zeros([3,3,3])[0][0] yields just an array   (array^1)

        # so it is important how we order the array
        # opening a single tiff file reveales that the axes are like 
        #   npTiffFileArray = tifffile.imread(self.tiffFileTuple[0])
        #   (yLenght, xLength) = np.shape(npTiffFileArray)
        #   e.g. npTiffFileArray[0] yields set of pixels along the x-axis
        #
        #   so we will need to adopt the convention:
        #   (length_Z, length_Y, length_X) = np.shape( aArray )
        #   aArray[ z0:z1, y0:y1, x0:x1]


        def getSubarray(sliceList : tuple) -> np.array:
            # use a tuple(slices) to build an (in memory) array 
            # from the contents of the tiff files that we access from the hard drive
            subarrayDimensions = [len(range(0,self.shape[i])[aSlice]) for i, aSlice in enumerate(sliceList)]

            subArray = np.empty(subarrayDimensions, dtype=self.dtype )

            for subArrayIndex , zarrListIndex in enumerate(range(0,len(self.zarrFileTuple))[sliceList[0]]): 
                subArray[subArrayIndex,:,:] = self.zarrFileTuple[zarrListIndex][sliceList[1:]]

            return subArray

        # presume that the single element of the zarrFileTuple is a 3d tiff
        if len(self.zarrFileTuple) == 1: 
            return self.zarrFileTuple[0][idx]


        if isinstance(idx,int): 
            return self.zarrFileTuple[idx][:,:]

        if isinstance(idx,slice):
            # Maybe this here should reaturn a custom 'tiffstackarray' itself?
            # Otherwise, if we have a very large tiff stack, 
            # we are likely using this class here to return subsets of the tiffstack as arrays
            # because keeping the full tiffstack in memory might not be feasible
            # and with this we can very easily return the full tiffstack as an array, just by calling
            # tiffStackArray[:]
            # 
            # Though on the other hand, this might be an elegant-ish way to cast a 
            # subset of a tiffstack to an array. So let's do the latter for now.
            #pass

            sliceList = tuple([idx] + [slice(0, dimension) for dimension in self.shape[1:] ])
            subArray = getSubarray(sliceList)

            return subArray
        
        # if we have a list of tuple of slices:
        if all([isinstance(x,slice) for x in idx ]):

            #subarrayDimensions = [len(range(0,self.shape[i])[aSlice]) for i, aSlice in enumerate(idx)]
            #subArray = np.empty(subarrayDimensions, dtype=self.dtype )
            #for subArrayIndex , zarrListIndex in enumerate(range(0,len(self.zarrFileTuple))[idx[0]]): 
            #    subArray[subArrayIndex,:,:] = self.zarrFileTuple[zarrListIndex][idx[1:]]

            subArray = getSubarray(idx)

            return subArray

        raise Exception("idx was neither 'int' nor 'slice' nor tuple(slice).")

    #def SliceIndexArrays

    def openTiffsAsZarrs(self,tiffFileTuple : tuple) -> list:
        
        zarrFileList = []

        for file in tiffFileTuple:

            zarrTiffStore = imread(file, aszarr=True)
            zarrImage = zarr.open(zarrTiffStore, mode='r')
            zarrFileList.append(zarrImage)

        return tuple(zarrFileList)

    def getTiffFileTuple(self, imagePath : str , fileType : str) -> tuple:
        # get all the tiff files in a given folder

        if isinstance(imagePath,str): 
            imagePath = os.path.abspath( imagePath )
        else: raise Exception("'imagePath' is not of type 'str', it is if type %s" %(str(type(imagePath))))

        if isinstance(imagePath,str) and os.path.isfile(imagePath):
            return tuple([imagePath])
        
        # otherwise we presume to have the path to a folder with multiple tiff files
        tiffFileList = []
        for file in sorted(os.listdir(imagePath)):

            # use regular expressions to be specific and flexible in defining file endings
            matchedFileTypes = re.findall(fileType,file)
            if len(matchedFileTypes) == 0 : continue # file did not match our fileType

            if file.endswith(matchedFileTypes[-1]):
                # double check that the regular expression match actually occured at the end of the file
                tiffFileList.append( os.path.join(imagePath,file))

        if len(tiffFileList) == 0 : 
            raise Exception("No files found whose ending matches '%s' in \n%s" %(fileType, imagePath))

        return tuple(tiffFileList)



if __name__ == "__main__":

    imageFilePath = "/media/ssdshare1/general/computational_projects/brain_segmentation/DaeHee_NeuN_data/20190621_11_58_27_349_fullbrain_488LP30_561LP140_642LP100/Ex_2_Em_2_destriped_stitched_master"
    #imageFilePath = "../NeuNBrainSegment_compressed.tiff"
    
    myTiffStack = tiffStackArray(imageFilePath)

    array1 = myTiffStack[0]

    array2 = myTiffStack[20:22]

    array3 = myTiffStack[ 10:20, 30:40, 50:60 ]

    #array4 = myTiffStack[:]


    #virtual_memory().total
    print("Done!")
