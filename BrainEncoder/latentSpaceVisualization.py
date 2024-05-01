import torch
import numpy as np

import sklearn.manifold as manifold
import sklearn.cluster  as cluster

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import time

def getLatentSpaceVectors(model, DEVICE, data_loader, saveAs = None):

    latentSpaceVectors = []
    labelList=[]

    model.eval() # Switch the model to evaluation mode

    with torch.no_grad():
        with torch.cuda.amp.autocast():

            for i, batch in enumerate(data_loader):

                imagesOnDevice = batch[0].to(DEVICE)
                labels = batch[1]


                #model.eval()  # Set the model to evaluation mode
                latentSpaceTensor = model.encoder(imagesOnDevice)

                latentSpaceArray = latentSpaceTensor.cpu().numpy()

                nLatendSpaceVectors = np.shape(latentSpaceArray)[0]

                latentSpaceList = [latentSpaceArray[i,:] for i in range(0, nLatendSpaceVectors ) ]

                latentSpaceVectors.extend(latentSpaceList)

                labelsArray = labels.cpu().numpy()

                label_n_dimensions = len( np.shape(labelsArray))

                labelTempList = np.split(labelsArray,nLatendSpaceVectors,axis=0)
                # in case we have labels that are multidimensional, we don't was a list 2d arrays
                if label_n_dimensions > 1: labelTempList = [ x[0] for x in labelTempList ]

                labelList.extend(labelTempList)

                #if len(latentSpaceVectors) > 10000: break




    if saveAs is not None:

        np.save( saveAs+"_latentSpaveVectors.npy", np.asarray(latentSpaceVectors))
        np.save( saveAs+"_labels.npy"            , np.asarray(labelList))

    return     latentSpaceVectors , labelList

def plotLatentSpaceVectors(latentSpaceVectors, nClusters = None, doShow = True):

    n_components = 2
    TSNE = manifold.TSNE(n_components)

    latentSpaceArray = np.asarray(latentSpaceVectors)

    tsne_embedding = TSNE.fit_transform(latentSpaceArray)

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

    return None



if __name__ == "__main__":
    import re



    inputModel = "../BrainEncoder_LD256.pth"

    import BrainEncoder as BrainEncoder
    from ImageSubvolumeDataset import ImageSubvolumeDataset


    # Checking is CUDA available on current machine
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Project running on device: ", DEVICE)

    # configurations for the task
    config = {"batchSize": 10}

    latentSpaceDimensionality = int(re.search("\d+",inputModel).group())

    model = BrainEncoder.AutoEncoder( latentSpaceSize=latentSpaceDimensionality).to(DEVICE)

    model.load_state_dict(torch.load(inputModel))

    subvolumeSize = 32

    dataset = ImageSubvolumeDataset("../NeuNBrainSegment_compressed.tiff", subvolumeSize = subvolumeSize)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                              shuffle=False, collate_fn=dataset.collate_array)
    
    latentSpaceVectors , labelList = getLatentSpaceVectors(model, DEVICE, data_loader)

    plotLatentSpaceVectors(latentSpaceVectors, nClusters = 3)

    print("Done!")
    
