import torch

import BrainEncoder.BrainEncoder as BrainEncoder

import torchvision # for handling input data
from torchvision import transforms # to manipulate input data

# torchsummary for easy checking and debugging
from torchinfo import summary


import time # calculate elapsed time

import numpy as np

def train(model, dataloader, criterion, optimizer):
    
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

def collate_2d_tiffs(batch):

    def processImage(image):

        image = image.convert("L") # turn image into single color

        process = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize images to the same size as FashionMNIST
            transforms.ToTensor(), 
            transforms.Pad([2])
            ])

        if len(np.shape(image)) >2: # turn 3d images into 2d ones

            smallestDimLength= min(np.shape(image))
            smallest_Dim_index = np.shape(image).index(smallestDimLength)
            
            image =  np.sum(np.asarray((image)),smallest_Dim_index)

        return process(image)

    imageBatchList = [processImage(data[0]) for data in batch]

    imageBatchTensor = torch.concat(imageBatchList).unsqueeze(1) # should have size [batchSize,1,imageXDim, imageYDim]

    # labels, note that we should convert the labels to LongTensor
    labelTensor = torch.LongTensor([data[1] for data in batch])

    return imageBatchTensor, labelTensor




if __name__ == "__main__":


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
    dataset = torchvision.datasets.ImageFolder(root='NeuNBrainSegment_compressed_2d_tif' )#, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size= config["batchSize"], shuffle=True, collate_fn=collate_2d_tiffs)


    model = BrainEncoder.AutoEncoder().to(DEVICE)

    ### Prep Training
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    print(summary(model))


    for i in range(config["epochs"]):
        epochStartTime = time.time()

        curr_lr = float(optimizer.param_groups[0]["lr"])
        train_loss = train(model, train_loader, criterion, optimizer)
        
        epochTimeElapsed = time.time()-epochStartTime

        print(f"Epoch {i+1}/{config['epochs']}\nTrain loss: {train_loss:.2E}\t lr: {curr_lr:.2E}, epoch time: {epochTimeElapsed:.1f} s'")


    print("done!")