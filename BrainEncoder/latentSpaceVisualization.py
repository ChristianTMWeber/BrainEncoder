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


    silhouetteScore = metrics.silhouette_score( latentSpaceArray , kmeans_labels, metric='euclidean')

    print( "K-means cluster with k=%i: Silhouette score = %f" %(nClusters, silhouetteScore))

    return kmeans_labels


def colorImageByClusterIndex(imageNPArray, latentSpaceArray,clusterIndexArray, labelArray, subvolumeSize, clusterK):

    score = metrics.silhouette_score( latentSpaceArray , clusterIndexArray, metric='euclidean')

    # (34, 34, 34), ### one color that I had removed from the list
    kelly_colors = [(242, 243, 244), (243, 195, 0), (135, 86, 146), (243, 132, 0), (161, 202, 241), (190, 0, 50), (194, 178, 128), (132, 132, 130), 
                    (0, 136, 86), (230, 143, 172), (0, 103, 165), (249, 147, 121), (96, 78, 151), (246, 166, 0), (179, 68, 108), (220, 211, 0), (136, 45, 23), 
                    (141, 182, 0), (101, 69, 34), (226, 88, 34), (43, 61, 38), 
                    ] 

    alphaArray = np.sum(imageNPArray[0:0+subvolumeSize,:,:], axis=0)
    alphaArray  = (alphaArray-np.min(alphaArray)) / (np.max(alphaArray)-np.min(alphaArray)) * 255 * 50
    alphaArray  = alphaArray.astype(np.uint8)

    clusterColorArray = np.zeros( list(np.shape(alphaArray))+[3] , dtype=np.uint8)
    nonEmptyArray = np.zeros( list(np.shape(alphaArray))+[3] , dtype=np.uint8)

    for label, clusterID in zip(labelArray, clusterIndexArray):

        if label[0] >0: continue

        sliceTuple = (slice(label[1],label[1]+subvolumeSize), slice(label[2],label[2]+subvolumeSize), slice(None))

        clusterColorArray[sliceTuple] = kelly_colors[clusterID]
        nonEmptyArray[sliceTuple] = 255


    alpha_image = Image.fromarray(alphaArray, 'L')
    image = Image.fromarray(clusterColorArray)
    image.putalpha(alpha_image)
    image.save('ImageClusterColord_k=%i.png' %(clusterK))  

    nonEmptyImage = Image.fromarray(nonEmptyArray)
    nonEmptyImage.save('nonEmptySubvolumes.png')  

    imageClusterColorsOnly = Image.fromarray(clusterColorArray)
    imageClusterColorsOnly.save('clusterColorsOnly_k=%i.png' %(clusterK))  

    alpha_image = Image.fromarray(alphaArray, 'L')
    alpha_image.save('BrainImage.png')  


    tifffile.imwrite('BrainImage.tiff', alphaArray)


    return None



def outputVolume(imageArray, nonEmptySlices, outputFileName = None):
    # visualize elements of the volume that we consider non-empty

    outputArray = np.zeros( np.shape(imageArray), dtype=np.uint8)

    for sliceTuple in nonEmptySlices:

        outputArray[sliceTuple] = 255


    if outputFileName is not None: 
        tifffile.imwrite(outputFileName, outputArray, compression='zlib')
    
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

    #imageFilePath = "/media/ssdshare1/general/computational_projects/brain_segmentation/DaeHee_NeuN_data/20190621_11_58_27_349_fullbrain_488LP30_561LP140_642LP100/Ex_2_Em_2_destriped_stitched_master"
    #dataset = ImageSubvolumeDataset(imageFilePath, subvolumeSize = subvolumeSize, minimumFractionalFill= 1E-4,
    #                                regionOfRelevance=(slice(1000,1064), slice(2000,2000+3000),slice(950,900+2700)),
    #                                batchSize = config["batchSize"], randomBatches = False ,
    #                                nAugmentedSamples = 0 )
    #                                #regionOfRelevance=(slice(1000,1064), slice(2000,2000+3000),slice(950,900+2700)))
    #                                #regionOfRelevance=(slice(1000,1064), slice(0,4500),slice(0,3500)))


    imageFilePath = "NeuNBrainSegment_compressed.tiff"
    dataset = ImageSubvolumeDataset(imageFilePath, subvolumeSize = subvolumeSize, minimumFractionalFill= 1E-4,
                                    regionOfRelevance=(slice(None)),
                                    batchSize = config["batchSize"], randomBatches = False ,
                                    nAugmentedSamples = 0 )
    

    outputVolume(dataset.imageNPArray, dataset.subvolumeSlices, outputFileName = "nonEmptySubvolumes.tiff")
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                              shuffle=False, collate_fn=dataset.collate_array)
    
    latentSpaceVectors , labelList = getLatentSpaceVectors(model, DEVICE, data_loader)
    latentSpaceArray = np.asarray(latentSpaceVectors)[:,0:8]


    # normalize the latent space vectors
    #latentSpaceVectors = [ x / np.linalg.norm(x) for x in latentSpaceVectors]

    for nClusters in range(2,11):

        plotName = "LatenSpaceVisualization_LD%i_ncluster_%s.png" %(latentSpaceDimensionality, str(nClusters).zfill(2))

        clusterIndexArray =plotLatentSpaceVectors(latentSpaceArray, nClusters = nClusters, 
                            outputFileName = plotName)
        
        colorImageByClusterIndex(dataset.imageNPArray, latentSpaceArray,clusterIndexArray, labelList,subvolumeSize,nClusters)


    print("Done!")
    
