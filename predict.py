import argparse
import utils


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--image_path',type=str,default='./flowers/valid/19/image_06158.jpg')
args = parser.parse_args()

#load the model
model, optimizer, criterion = utils.load_checkpoint('checkpoint.pth')

#predict
topk =5
print('predicting image ', args.image_path)
probabilities, classes = utils.predict(args.image_path, model, topk)

#print prediction results
for i in range(topk):
    print("Class - {} ; Probability - {} ".format(classes[i], round(float(probabilities[i])*100,2)))
