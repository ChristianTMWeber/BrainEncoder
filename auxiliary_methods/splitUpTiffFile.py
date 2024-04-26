import numpy as np
import tifffile
import matplotlib.pyplot as plt

import os

import PIL

import re


def segmentIsTooEmpty(segment: "np.array", emptyThreshold = 0.001) -> "bool":


    fractionalFill = np.sum( segment > 0 ) / np.prod(np.shape(segment))

    return fractionalFill < emptyThreshold


if __name__ == '__main__':

    inputFileName = "..\\NeuNBrainSegment_compressed.tiff"


    outputFolder = "..\\NeuNBrainSegment_compressed_2d_tif\\brain"

    os.makedirs(outputFolder, exist_ok = True) 


    output_2d_image = "2d" in outputFolder
   

    mainImage = tifffile.imread(inputFileName)


    # Define the size of each segment
    segment_size = 50

    segmentCounter = 0
    emptySegmentCounter = 0

    # Iterate over each segment
    for i in range(0, mainImage.shape[0], segment_size):
        for j in range(0, mainImage.shape[1], segment_size):
            for k in range(0, mainImage.shape[2], segment_size):

                segmentCounter +=1
                # Extract the current segment
                segment = mainImage[i:i+segment_size, j:j+segment_size, k:k+segment_size]

                if segmentIsTooEmpty(segment, emptyThreshold = 0.001):
                    emptySegmentCounter +=1
                    continue

                if output_2d_image: segment = np.average(segment,2)
                
                # Specify the file path for the TIFF file
                outputFileName = f"segment_{i}_{j}_{k}.tiff"

                full_file_path = os.path.join(outputFolder,outputFileName)
                

                if output_2d_image: 

                    segment *= 255 # PIL "L" option is saved as uint8, 0 to 255, let's convert to that range

                    image = PIL.Image.fromarray(segment)

                    image = image.convert("L")

                    image.save(full_file_path)


                else: # PIL can only write 2d tiffs, so for 3d, rely on tifffile

                    segment *= 255 

                    segment = segment.astype(np.uint8)

                    tifffile.imwrite(full_file_path, segment)#, compression='zlib')


    print( "Total segments %i" %segmentCounter)
    print( "Too empty segments %i" %emptySegmentCounter)
    print( "Segment yield %.3f" %( (segmentCounter-emptySegmentCounter)/segmentCounter))

    ### Code that I had used to turn image_subvolume.npy into tiff
    #
    #data = np.load('.\\mira_image_files\\2_Subvolume-files-data\\image_subvolume.npy')
    #tifffile.imwrite("NeuNBrainSegment_compressed.tiff", data, compression='zlib')
    #series1 = tifffile.imread('NeuNBrainSegment_compressed.tiff')
    #
    #import pdb; pdb.set_trace()