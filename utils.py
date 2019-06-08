import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    #Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir,transform=data_transforms['train'])
    val_datasets = datasets.ImageFolder(valid_dir,transform=data_transforms['valid'])
    test_datasets = datasets.ImageFolder(test_dir,transform=data_transforms['test'])

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets,batch_size=32,shuffle=True)
    valloader = torch.utils.data.DataLoader(val_datasets,batch_size=32,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets,batch_size=32,shuffle=True)

    return train_datasets, trainloader, valloader, testloader


#function to decide GPU vs CPU
def determine_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('GPU used for training')
    else:
        device = torch.device("cpu")
        print('CPU used for training')
    return device

def create_model():
    # define the model
    model = models.vgg19(pretrained=True)

    #freeze params
    for param in model.parameters():
        param.requires_grad = False

    #define classifier
    classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features,2048),
                          nn.ReLU(),
                          nn.Dropout(p=0.15),
                          nn.Linear(2048, 1024),
                          nn.ReLU(),
                          nn.Dropout(p=0.15),
                          nn.Linear(1024, 102),
                          nn.LogSoftmax(dim=1))
    #assign classifier
    model.classifier = classifier
    #determinde device
    device = determine_device()
    model.to(device)

    #initialize criterian and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)
    #optimizer = optim.SGD(model.classifier.parameters(),lr = 0.001)

    return model, criterion, optimizer , device


# validation function
def validation(model,dataloader,criterion, device):
    model.eval()
    validation_loss = 0
    accuracy =0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        # Loss calculation
        validation_loss += criterion(output,labels)

        # Accuracy calculation
        output = torch.exp(output)
        equality = (labels.data == output.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return validation_loss/len(dataloader), accuracy/len(dataloader)


def train_model(model, criterion, optimizer, train_dataloader, valid_dataloader, device, epochs):
    print_break = 50

    for epoch in range(epochs):
        train_loss = 0
        step = 0
        for images, labels in train_dataloader:
            step += 1
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()

            # Forward images
            output = model(images)

            # Loss
            loss = criterion(output,labels)
            train_loss += loss.item()

            # Update model for train
            loss.backward()
            optimizer.step()

            # Validate
            if step%print_break==0:
                val_accuracy = 0
                val_loss = 0
                with torch.no_grad():
                    model.eval()
                    val_loss, val_accuracy = validation(model,valid_dataloader,criterion,device)
                    print('Step: {}; Validation Loss: {}; Validation Accuracy: {}'.format(step,val_loss,val_accuracy))
        print('Epoch: {}/{}; Training Loss: {}'.format(epoch+1,epochs,train_loss/len(train_dataloader)))
    print("training complete")

# Save the checkpoint
def save_checkpoint(model, criterion, optimizer, train_dataset):
    # Store class_to_idx into a model property
    model.class_to_idx = train_dataset.class_to_idx

    print ('Saving checkpoint')
    checkpoint = {
        'model':model,
        'classifier': model.classifier,
        'arch': 'vgg19',
        'optimizer':optimizer,
        'optimizer_state_dict':optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'epochs':'10',
        'criterion':criterion
        }

    torch.save(checkpoint, 'checkpoint.pth')
    #print("model: ", model)
    #print(" Model state dict keys: ", model.state_dict().keys())
    print ('checkpoint saved')


#function to load the model from check point
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = checkpoint['criterion']
    epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']

    return model, optimizer, criterion

#function to process the image
def process_image(path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(path)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                          std=(0.229, 0.224, 0.225))
                                    ])
    pil_image = image_transforms(pil_image)
    return pil_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model = model.to('cpu')

    with torch.no_grad():
        output = model.forward(process_image(image_path).unsqueeze_(0))
        probabilities, classes = torch.topk(output, topk)
        probabilities = probabilities.exp()
    #print("1: probabilities:{}, classes:{} ", probabilities,classes)

    probabilities, classes = np.array(probabilities[0].detach().numpy()), np.array(classes[0].detach().numpy())
    classes = classes.astype('str')
    #print("2: probabilities:{}, classes:{} ", probabilities,classes)

    classes = np.array([model.class_to_idx[i] for i in classes])[probabilities.argsort()]
    probabilities = np.sort(probabilities)
    #print("3: probabilities:{}, classes:{} ", probabilities,classes)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    labels = list(cat_to_name.values())
    classes = [labels[x] for x in classes]
    #print("4: probabilities:{}, classes:{} ", probabilities,classes)

    return probabilities, classes
