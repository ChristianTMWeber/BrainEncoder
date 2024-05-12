import torch

import BrainEncoder.BrainEncoder as BrainEncoder

from BrainEncoder.ImageSubvolumeDataset import ImageSubvolumeDataset


import matplotlib.pyplot as plt

import numpy as np
import re

def plotTensor( inTensor, outTensor, outputFileName = None, labels = None, doShow = True):

    def prepTensor(aTensor):

        tensorCPU = aTensor.cpu() # convert tensor to CPU so that we can turn it into a np array
        
        # sum output along 5th axis in preparation to make 2d images
        # remember, components of the tensor are
        # tensor[batch, channel, X, Y, Z]
        tensorZSummed = torch.sum( tensorCPU, dim=4) 

        npTensor = tensorZSummed.numpy()

        return npTensor
    
    def addToAxes(axesObj, col, row, array_2d, label):

        labelNoTensor = [x.item() for x in label]

        # using item method as 'label' should be a tensor
        xLabelString = "x=%i \ny=%i \nz=%i" %(label[2].item(), label[1].item(), label[0].item())


        axesObj[row, col].imshow(array_2d, cmap='gray')  # Display as grayscale
        axesObj[row, col].axis('on')  # keep axis
        axesObj[row, col].set_xticks([])
        axesObj[row, col].set_yticks([])
        axesObj[row, col].set_xlabel(xLabelString, fontsize=6)
        return axesObj

    nColumns = inTensor.size()[0]
    
    if labels is None:   labels = [""]*nColumns

    #                        n_rows,   n_cols
    fig, axes = plt.subplots(     2, nColumns, figsize=(10, 5))

    inTensorNP  = prepTensor(inTensor)
    outTensorNP = prepTensor(outTensor)

    for indx in range(0, nColumns): 
        addToAxes(axes, indx, 0,  inTensorNP[indx,0,:,:], labels[indx])
        addToAxes(axes, indx, 1, outTensorNP[indx,0,:,:], labels[indx])

    axes[0, 0].set(ylabel="Input Image")
    axes[1, 0].set(ylabel="AE output")

    #plt.tight_layout()
    if doShow: plt.show()
    if outputFileName is not None: plt.savefig(outputFileName)

    return fig, axes

def showModelVerisimilitude(model, aDataloader, nTensorsToPlot = None, 
                            plotName = None, showPlot = False, 
                            skipNBatches = 0):

    # Checking is CUDA available on current machine
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataiter = iter(aDataloader)

    for x in range(0,skipNBatches): next(dataiter)
    images, labels = next(dataiter)

    if nTensorsToPlot is not None:
        images = images[0:nTensorsToPlot]

    imagesOnDevice = images.to(DEVICE)

    # Get reconstructed images from the autoencoder
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            #model.eval()  # Set the model to evaluation mode
            outputs = model(imagesOnDevice)

    fig, axes = plotTensor( imagesOnDevice, outputs, labels = labels, 
               outputFileName = plotName, doShow = showPlot)

    del imagesOnDevice, outputs

    return fig, axes

if __name__ == "__main__":

    inputModel = "BrainEncoder_LD256.pth"

    # Checking is CUDA available on current machine
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Project running on device: ", DEVICE)

    # configurations for the task
    config = {"batchSize": 10}

    #                       1 or more digits after the string "LD"
    latentSpaceDimensionality = int(re.search("(?<=LD)\d+",inputModel).group())

    model = BrainEncoder.AutoEncoder( latentSpaceSize=latentSpaceDimensionality).to(DEVICE)
    # Load the trained model weights

    model.load_state_dict(torch.load(inputModel))
    

    model.eval()  # Set the model to evaluation mode

    # Create a dataset from a folder containing images

    subvolumeSize = 32

    imageFilePath = "/media/ssdshare1/general/computational_projects/brain_segmentation/DaeHee_NeuN_data/20190621_11_58_27_349_fullbrain_488LP30_561LP140_642LP100/Ex_2_Em_2_destriped_stitched_master"
    #imageFilePath = "NeuNBrainSegment_compressed.tiff"

    dataset = ImageSubvolumeDataset(imageFilePath, subvolumeSize = subvolumeSize, minimumFractionalFill= 1E-4,
                                    regionOfRelevance=(slice(1000,1064), slice(0,4500),slice(0,3500)))


    data_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                                shuffle=True, collate_fn=dataset.collate_array)
    

    showModelVerisimilitude(model, data_loader, plotName ="BrainEncoder_visualization.png")

    print("done!")