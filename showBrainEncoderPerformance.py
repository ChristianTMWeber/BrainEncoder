import torch
import torchvision

import BrainEncoder.BrainEncoder as BrainEncoder

from BrainEncoder.ImageSubvolumeDataset import ImageSubvolumeDataset

from trainBrainEncoder import collate_array

import matplotlib.pyplot as plt

import numpy as np
import re

def plotTensor( inTensor, outTensor, outputFileName = None, doShow = True):

    def prepTensor(aTensor):

        tensorCPU = aTensor.cpu() # convert tensor to CPU so that we can turn it into a np array
        
        # sum output along 5th axis in preparation to make 2d images
        # remember, components of the tensor are
        # tensor[batch, channel, X, Y, Z]
        tensorZSummed = torch.sum( tensorCPU, dim=4) 

        npTensor = tensorZSummed.numpy()

        return npTensor
    
    def addToAxes(axesObj, col, row, array_2d):

        axesObj[row, col].imshow(array_2d, cmap='gray')  # Display as grayscale
        axesObj[row, col].axis('off')  # Hide axes for a cleaner display

        return axesObj

    nColumns = inTensor.size()[0]

    #                        n_rows,   n_cols
    fig, axes = plt.subplots(     2, nColumns, figsize=(10, 6))

    inTensorNP  = prepTensor(inTensor)
    outTensorNP = prepTensor(outTensor)

    for indx in range(0, nColumns): 
        addToAxes(axes, indx, 0,  inTensorNP[indx,0,:,:])
        addToAxes(axes, indx, 1, outTensorNP[indx,0,:,:])

    plt.tight_layout()
    if doShow: plt.show()
    if outputFileName is not None: plt.savefig(outputFileName)

    return fig, axes

if __name__ == "__main__":

    inputModel = "BrainEncoder_LD1024.pth"

    # Checking is CUDA available on current machine
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Project running on device: ", DEVICE)

    # configurations for the task
    config = {"batchSize": 10}

    latentSpaceDimensionality = int(re.search("\d+",inputModel).group())

    model = BrainEncoder.AutoEncoder( latentSpaceSize=latentSpaceDimensionality).to(DEVICE)
    # Load the trained model weights

    model.load_state_dict(torch.load(inputModel))
    

    model.eval()  # Set the model to evaluation mode

    # Create a dataset from a folder containing images

    subvolumeSize = 32

    dataset = ImageSubvolumeDataset("NeuNBrainSegment_compressed.tiff", subvolumeSize = subvolumeSize)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                                shuffle=False, collate_fn=collate_array)
    
    # Load one batch of test images
    dataiter = iter(data_loader)

    images, _ = next(dataiter)
    imagesOnDevice = images.to(DEVICE)
    

    
    # Get reconstructed images from the autoencoder
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()  # Set the model to evaluation mode
            outputs = model(imagesOnDevice)


    plotTensor( imagesOnDevice, outputs)


    print("done!")