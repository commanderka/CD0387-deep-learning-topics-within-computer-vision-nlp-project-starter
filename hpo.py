#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
from PIL import ImageFile
import boto3
import logging
import sys
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device):
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    with torch.no_grad():
        running_loss=0
        correct=0
        for data, target in test_loader:
             data=data.to(device)
             target=target.to(device)
             pred = model(data)             #No need to reshape data since CNNs take image inputs
             loss = criterion(pred, target)
             running_loss+=loss
             pred=pred.argmax(dim=1, keepdim=True)
             correct += pred.eq(target.view_as(pred)).sum().item()
    print(f"Test Loss {running_loss/len(test_loader.dataset)}, \
         Test Accuracy {100*(correct/len(test_loader.dataset))}%")
    
def train(model, train_loader, criterion, optimizer, device, epochs=1):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    model.train()
    for e in range(epochs):
     running_loss=0
     correct=0
     for data, target in train_loader:
         data=data.to(device)
         target=target.to(device)
         optimizer.zero_grad()
         pred = model(data)             #No need to reshape data since CNNs take image inputs
         loss = criterion(pred, target)
         running_loss+=loss
         loss.backward()
         optimizer.step()
         pred=pred.argmax(dim=1, keepdim=True)
         correct += pred.eq(target.view_as(pred)).sum().item()
     print(f"Epoch {e}: Train Loss {running_loss/len(train_loader.dataset)}, \
         Train Accuracy {100*(correct/len(train_loader.dataset))}%")


def net(n_classes=10):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, n_classes))
    
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass


def main(args):
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()

    transform_comp = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_train = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir,"train"),transform=transform_comp)
    dataset_val = torchvision.datasets.ImageFolder(root=os.path.join(args.data_dir,"valid"),transform=transform_comp)

    train_dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=args.test_batch_size)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net(n_classes=len(dataset_train.classes))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    train(model, train_dataloader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, val_dataloader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''

    save_model(model,args.model_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    '''
    TODO: Specify any training args that you might need
    '''

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    args=parser.parse_args()
    
    main(args)
