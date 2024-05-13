import torch

# torchsummary for easy checking and debugging
from torchinfo import summary


import time # calculate elapsed time

import numpy as np

import BrainEncoder.BrainEncoder as BrainEncoder
from BrainEncoder.ImageSubvolumeDataset import ImageSubvolumeDataset

from BrainEncoder.multi_similarity_loss import MultiSimilarityLoss

from showBrainEncoderPerformance import showModelVerisimilitude


def subvolumeDistance(labelA, labelB, subvolumeSize):

    # sqrt( dX^2 + dY^2 + dz^2); dX = x1-x2
    euclidianDistance = np.sqrt(np.sum(np.square(np.asarray(labelA)-np.asarray(labelB)))) 

    return euclidianDistance / subvolumeSize


def train(model, dataloader, criterion, optimizer,scaler, subvolumeSize):
    
    model.train()


    train_loss = 0.0

    RecoLoss_ledger = 0.0
    MultiSimLoss_ledger = 0.0 # MultiSimLoss - MultiSim

    MultiSimLossMultiplier = 1.E+3 
    # Multiply MultiSimilarityLoss with this constant
    # Otherwise one loss might be much larger than the other
    # and we end up optimizing only one of the two losses

    multiSimilarityLossCalculator = MultiSimilarityLoss()

    for i, batch in enumerate(dataloader):

        subvolumes = batch[0]
        batchSize= subvolumes.size(0)
        nVoxelsInSubvolume = np.prod(subvolumes.size()[2:]) # = subvolumeSize**3
        labelTensor = batch[1]

        maxDistanceForSameFeatures = 2 + 1E-5

        

        similarLabelToFirstBatchElement= torch.LongTensor(
            [subvolumeDistance(labelTensor[0], label, subvolumeSize) < maxDistanceForSameFeatures 
             for label in labelTensor] )
        
        #locationIDList = []
        #for tocationTuple in labelTensor:
        #    locationIDList.append( np.sum(np.asarray(tocationTuple) * np.shape(dataloader.imageNPArray)))
        #
        #similarLabelToFirstBatchElement= torch.LongTensor(locationIDList)

        
        optimizer.zero_grad()
        x = batch[0].to(DEVICE)
        
        # Here we implement the mixed precision training
        with torch.cuda.amp.autocast():
            y_recons = model(x)
            latentSpaceTensor = model.encoder(x)
            RecoLoss = criterion(y_recons, x)


        latentSpaceTensorNormalized = torch.nn.functional.normalize(latentSpaceTensor,dim=1)
        latentSpaceTensorNormalized = latentSpaceTensorNormalized.to(torch.float32)

        MultiSimLoss = multiSimilarityLossCalculator.forward(latentSpaceTensorNormalized, similarLabelToFirstBatchElement)
        
        # not clear at the moment that this is the best way to normalize the MS loss by the number of positive samples
        nPositiveSamples = similarLabelToFirstBatchElement.sum().item() + 1E-5 # add small epsilon so that this can not yield NaN or Inf
        
        RecoLoss_normalized = RecoLoss/(batchSize*nVoxelsInSubvolume)

        MultiSimLoss_normalized = MultiSimLoss/( batchSize * nPositiveSamples  )

        RecoLoss_ledger += RecoLoss_normalized.item()
        MultiSimLoss_ledger += MultiSimLoss_normalized.item() * MultiSimLossMultiplier

        commbinedNormalizedLoss = RecoLoss/(batchSize*nVoxelsInSubvolume) + \
            MultiSimLossMultiplier * MultiSimLoss/( batchSize * nPositiveSamples  )
        #commbinedNormalizedLoss = RecoLoss/(batchSize*nVoxelsInSubvolume) 
        

        # RecoLoss inclues in part this property:  grad_fn=<MseLossBackward0>
        # While MultiSimLoss includes: grad_fn=<DivBackward0>
        # and commbinedNormalizedLoss includes:   grad_fn=<AddBackward0>
        # let's keep this in mind for later to check if what relevance this has


        if np.isnan(commbinedNormalizedLoss.item()):
            print("Observed NaN.")
            #import pdb; pdb.set_trace()
            pass
            continue



        train_loss += commbinedNormalizedLoss.item()#   RecoLoss.item()

        #if np.isnan(train_loss):
        #    pass
        #    #import pdb; pdb.set_trace()
        
        scaler.scale(commbinedNormalizedLoss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    
        # remove unnecessary cache in CUDA memory
        torch.cuda.empty_cache()
        del x, y_recons, latentSpaceTensorNormalized, RecoLoss, RecoLoss_normalized, MultiSimLoss_normalized, latentSpaceTensor, MultiSimLoss

    print("RecoLoss_normalized = %.2E, MultiSimLoss_normalized = %.2E" %(RecoLoss_ledger, MultiSimLoss_ledger))

    
    train_loss /= len(dataloader)

    return train_loss

if __name__ == "__main__":

    print("Start")



    # Checking is CUDA available on current machine
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Project running on device: ", DEVICE)

    # configurations for the task
    config = {
        "batchSize": 128,
        "epochs": 300,
        "lr": 1e-4,   # learning rate
    }


    # Create a dataset from a folder containing images

    subvolumeSize = 32
    latentSpaceSize = 256

    # size is 4801, 9110, 7338
    imageFilePath = "/media/ssdshare1/general/computational_projects/brain_segmentation/DaeHee_NeuN_data/20190621_11_58_27_349_fullbrain_488LP30_561LP140_642LP100/Ex_2_Em_2_destriped_stitched_master"
    #imageFilePath = "NeuNBrainSegment_compressed.tiff"

    #dataset = ImageSubvolumeDataset(imageFilePath, subvolumeSize = subvolumeSize, minimumFractionalFill= 1E-4,
    #                                regionOfRelevance=(slice(1000,1000+subvolumeSize*40), slice(0,4500),slice(0,3500)))
    # check that this is z-slice is compatible with the limit on open files: ulimit -n
    # increase limit with: ulimit -n <limit>

    dataset = ImageSubvolumeDataset(imageFilePath, subvolumeSize = subvolumeSize, minimumFractionalFill= 1E-4,
                                    regionOfRelevance=(slice(1000,1064), slice(2000,2000+3000),slice(950,900+2700)),
                                    batchSize = config["batchSize"], randomBatches = True ,
                                    nAugmentedSamples = 64 )
                                    #regionOfRelevance=(slice(1000,1064), slice(0,4500),slice(0,3500)))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                                shuffle=True, collate_fn=dataset.collate_array)


    model = BrainEncoder.AutoEncoder(latentSpaceSize = latentSpaceSize).to(DEVICE)

    ### Prep Training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    print(summary(model, input_size=(1, 1, subvolumeSize, subvolumeSize, subvolumeSize)))



    for i in range(config["epochs"]):
        epochStartTime = time.time()

        curr_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train(model, dataset, criterion, optimizer,scaler, subvolumeSize)
        
        epochTimeElapsed = time.time()-epochStartTime

        #print(f"Epoch {i+1}/{config['epochs']}\nTrain loss: {train_loss:.2E}\t lr: {curr_lr:.2E}, epoch time: {epochTimeElapsed:.1f} s'")
        print(f"Epoch {i+1}/{config['epochs']}, Epoch time: {epochTimeElapsed:.1f} s'")

        torch.save(model.state_dict(), 'BrainEncoder_LD%i.pth' %latentSpaceSize)


    # Save the trained model
    torch.save(model.state_dict(), 'BrainEncoder_LD%i.pth' %latentSpaceSize)

    # save a comparison of some model inputs and outputs
    showModelVerisimilitude(model, dataset, nTensorsToPlot = 15,
                            plotName ='BrainEncoderVerisimilitude_LD%i.png' %latentSpaceSize)


    print("done!")