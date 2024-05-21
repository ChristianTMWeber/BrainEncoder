import torch
import numpy as np

import sklearn.manifold as manifold
import sklearn.cluster  as cluster
import sklearn.metrics as  metrics

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import skimage

from PIL import Image

import tifffile

def getLatentSpaceVectors(model, DEVICE, data_loader, saveAs = None):

    def extendListWithUnpackedTensor(aList,aTensor):


        anArray = aTensor.cpu().numpy()

        nElements = np.shape(anArray)[0]

        tempList = np.split(anArray,nElements,axis=0)

        if len( tempList[0] ) == 1: tempList = [ x[0] for x in tempList ]

        aList.extend(tempList)

        return tempList

    latentSpaceVectors = []
    labelList=[]

    #model.eval() # Switch the model to evaluation mode

    with torch.no_grad():
        with torch.cuda.amp.autocast():

            for i, batch in enumerate(data_loader):

                imagesOnDevice = batch[0].to(DEVICE)
                labels = batch[1]
                
                latentSpaceTensor = model.encoder(imagesOnDevice)


                _ = extendListWithUnpackedTensor(latentSpaceVectors,latentSpaceTensor)
                _ = extendListWithUnpackedTensor(labelList,labels)

                #if len(latentSpaceVectors) > 10000: break


    if saveAs is not None:

        np.save( saveAs+"_latentSpaveVectors.npy", np.asarray(latentSpaceVectors))
        np.save( saveAs+"_labels.npy"            , np.asarray(labelList))

    return latentSpaceVectors , labelList

def plotLatentSpaceVectors(latentSpaceArray, nClusters = None, doShow = True, outputFileName = None, 
                           TSNE_embedding_store = []):

    n_components = 2
    TSNE = manifold.TSNE(n_components)

    #latentSpaceArray = np.asarray(latentSpaceVectors)

    if len(TSNE_embedding_store) == 0: 
        tsne_embedding = TSNE.fit_transform(latentSpaceArray)
        TSNE_embedding_store.append(tsne_embedding) # cache the embedding, implement a better way later
    else:tsne_embedding=TSNE_embedding_store[0]

    dataDictForPandas = {'TSNE Variable 1': tsne_embedding[:,0],
                         'TSNE Variable 2': tsne_embedding[:,1],
                          }
    scatterHue = None

    

    if nClusters is not  None:

        kmeans = cluster.KMeans(init="k-means++", n_clusters= nClusters, n_init=4, random_state=0)

        kmeans.fit(latentSpaceArray)

        kmeans_labels = kmeans.predict(latentSpaceArray)

        # add the cluster labels to the dict, 
        dataDictForPandas['Cluster#']= kmeans_labels
        # and make sure we set the hue variables matched it the relevant dict key
        scatterHue = 'Cluster#'

    
    tsne_result_df = pd.DataFrame(dataDictForPandas)

    fig, ax = plt.subplots(1)
    sns.scatterplot(x='TSNE Variable 1', y='TSNE Variable 2', 
                    hue=scatterHue, data=tsne_result_df, ax=ax,s=120)
    if doShow: plt.show()
    if outputFileName is not None: plt.savefig(outputFileName)

    return None



if __name__ == "__main__":
    import re
    import os

    script_path = os.path.dirname(os.path.abspath(__file__))




    #inputModel = os.path.join(script_path,"../BrainEncoder_LD64_L2Pretrained.pth")
    inputModel = os.path.join(script_path,"../BrainEncoder_LD64_epoch040.pth")
    # 

    import BrainEncoder as BrainEncoder
    from ImageSubvolumeDataset import ImageSubvolumeDataset


    # Checking is CUDA available on current machine
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Project running on device: ", DEVICE)

    # configurations for the task
    config = {"batchSize": 10}

    latentSpaceDimensionality = int(re.search("(?<=LD)\d+",inputModel).group())

    model = BrainEncoder.AutoEncoder( latentSpaceSize=latentSpaceDimensionality).to(DEVICE)

    model.load_state_dict(torch.load(inputModel))

    subvolumeSize = 32

    dataset = ImageSubvolumeDataset("../NeuNBrainSegment_compressed.tiff", subvolumeSize = subvolumeSize)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                              shuffle=False, collate_fn=dataset.collate_array)
    
    latentSpaceVectors , labelList = getLatentSpaceVectors(model, DEVICE, data_loader)
    latentSpaceArray = np.asarray(latentSpaceVectors)[:,0:8]


    # normalize the latent space vectors
    #latentSpaceVectors = [ x / np.linalg.norm(x) for x in latentSpaceVectors]

    for nClusters in range(2,11):

        plotName = "LatenSpaceVisualization_LD%i_ncluster_%s.png" %(latentSpaceDimensionality, str(nClusters).zfill(2))

        plotLatentSpaceVectors(latentSpaceVectors, nClusters = nClusters, outputFileName = plotName)


    print("Done!")
    
