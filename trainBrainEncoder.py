import torch



import torchvision # for handling input data
from torchvision import transforms # to manipulate input data

# torchsummary for easy checking and debugging
from torchinfo import summary


import time # calculate elapsed time

import numpy as np

def train(model, dataloader, criterion, optimizer,scaler):
    
    model.train()
    train_loss = 0.0

    for i, batch in enumerate(dataloader):
        
        optimizer.zero_grad()
        x = batch[0].to(DEVICE)
        
        # Here we implement the mixed precision training
        with torch.cuda.amp.autocast():
            y_recons = model(x)
            loss = criterion(y_recons, x)
        
        train_loss += loss.item()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    
        # remove unnecessary cache in CUDA memory
        torch.cuda.empty_cache()
        del x, y_recons
    
    train_loss /= len(dataloader)

    return train_loss

def collate_array(batch):

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

    import BrainEncoder.BrainEncoder as BrainEncoder

    from BrainEncoder.ImageSubvolumeDataset import ImageSubvolumeDataset


    # Checking is CUDA available on current machine
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Project running on device: ", DEVICE)

    # configurations for the task
    config = {
        "batchSize": 64,
        "epochs": 3,
        "lr": 5e-4,   # learning rate
    }


    # Create a dataset from a folder containing images

    subvolumeSize = 32
    latentSpaceSize = 2048

    dataset = ImageSubvolumeDataset("NeuNBrainSegment_compressed.tiff", subvolumeSize = subvolumeSize)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], 
                                                shuffle=True, collate_fn=collate_array)


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
        train_loss = train(model, train_loader, criterion, optimizer,scaler)
        
        epochTimeElapsed = time.time()-epochStartTime

        print(f"Epoch {i+1}/{config['epochs']}\nTrain loss: {train_loss:.2E}\t lr: {curr_lr:.2E}, epoch time: {epochTimeElapsed:.1f} s'")



    # Save the trained model
    torch.save(model.state_dict(), 'BrainEncoder_LD%i.pth' %latentSpaceSize)

    print("done!")