# Imports
import json
import argparse
import utils

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,default='./flowers')
#parser.add_argument('--save_dir',type=str,default='checkpoint.pth')
#parser.add_argument('--arch',default='vgg19',type=str)
#parser.add_argument('--gpu',action='store_true',default=False)
#parser.add_argument('--learning_rate',default=0.001,type=float)
#parser.add_argument('--hidden_units',default=512,type=int)
parser.add_argument('--epochs',default=10,type=int)
args = parser.parse_args()

print('data directory: ', args.data_dir)
# initialize dataloaders
train_dataset, train_dataloader, valid_dataloader, test_dataloader = utils.load_data(args.data_dir)

#create and load model
model, criterion, optimizer, device = utils.create_model()

# train model
utils.train_model(model, criterion, optimizer, train_dataloader, valid_dataloader, device, args.epochs)

#test model
test_loss, test_accuracy = utils.validation(model,test_dataloader,criterion, device)
print('Test Loss: {}; Test Accuracy: {}'.format(test_loss, round(float(test_accuracy)*100,2)))

#save the check point
utils.save_checkpoint(model, criterion, optimizer, train_dataset)

