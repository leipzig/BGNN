from torch import nn
import torch
import progressbar
from earlystopping import EarlyStopping
import os
from torch.autograd import Variable
import csv
import time
from sklearn.metrics import confusion_matrix

CheckpointNameFinal = 'finalModel.pt'

accuracyFileName = "validation_accuracy.csv"
lossFileName = "training_loss.csv"
timeFileName = "time.csv"
epochsFileName = "epochs.csv"

import torchvision.models as models
def create_model(numberOfClasses, params):
    usePretrained = params["usePretrained"]
    model = None
    if usePretrained:
        print('using a pretrained resnet18 model...')
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, numberOfClasses)
    else:
        model = CNN(numberOfClasses, params)

    return model

# Build the convolutional Neural Network Class
class CNN(nn.Module):
    
    # Contructor
    def __init__(self, numberOfClasses, params):
        in_ch = params["n_channels"]
        n_channels = in_ch
        imageDimension = params["imageDimension"]
        kernels = params["kernels"]
        kernelSize = params["kernelSize"]
        
        super(CNN, self).__init__()
        i=0
        self.numOfLayers = len(kernels)
        self.numberOfClasses = numberOfClasses
        self.module_list = nn.ModuleList()
        
        out_ch = 0
        max_pool_kernel_size = 2
        padding = int((kernelSize-1)/2)
        
        while i < self.numOfLayers:
            out_ch = kernels[i]
            self.module_list.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernelSize, stride=1, padding=padding))
            self.module_list.append(nn.ReLU())
            self.module_list.append(nn.MaxPool2d(kernel_size=max_pool_kernel_size))
            in_ch = out_ch
            i = i + 1
        
        n_size = self._get_conv_output((n_channels, imageDimension, imageDimension))
        self.module_list.append(nn.Linear(n_size, numberOfClasses))
    
    # Prediction
    def forward(self, x):
        inpt, cnn_outputs, relu_outputs, maxpool_outputs, flattened_outputs = self.partial_forward(x)
        out = self.module_list[-1](flattened_outputs)
        return out
    
    
    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        x = Variable(torch.rand(bs, *shape))
        inpt, cnn_outputs, relu_outputs, maxpool_outputs, flattened_outputs = self.partial_forward(x)
        return flattened_outputs.shape[1]

    def partial_forward(self, x):
        cnn_outputs = []
        relu_outputs = []
        maxpool_outputs = []
        
        out = x
        inpt = out
        for i in range(self.numOfLayers):
            out = self.module_list[i*3](out)
            cnn_outputs.append(out)
            out = self.module_list[i*3+1](out)
            relu_outputs.append(out)
            out = self.module_list[i*3+2](out)
            maxpool_outputs.append(out)
        out = out.view(out.size(0), -1)
        flattened_outputs = out
        return inpt, cnn_outputs, relu_outputs, maxpool_outputs, flattened_outputs
    
        # Outputs in each steps
    def activations(self, x):
        #outputs activation this is not necessary
        inpt, cnn_outputs, relu_outputs, maxpool_outputs, flattened_outputs = self.partial_forward(x)
        out = self.module_list[-1](flattened_outputs)
        return inpt, cnn_outputs, relu_outputs, maxpool_outputs, flattened_outputs, out
      

def getModelFile(experimentName):
    return os.path.join(experimentName, CheckpointNameFinal)
    
def trainModel(train_loader, validation_loader, params, model, savedModelName):
    n_epochs = params["n_epochs"]
    patience = params["patience"]
    
    if not os.path.exists(savedModelName):
        os.makedirs(savedModelName)
    
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    training_loss_list=[]
    validation_accuracy_list=[]
    
    # early stopping
    early_stopping = EarlyStopping(path=savedModelName, patience=patience)

    print("Training started...")
    start = time.time()
    with progressbar.ProgressBar(maxval=n_epochs, redirect_stdout=True) as bar:
        bar.update(0)
        epochs = 0
        for epoch in range(n_epochs):
            criterion = nn.CrossEntropyLoss()
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    z = applyModel(batch["image"], model)
                    loss = criterion(z, batch["class"])
                    loss.backward()
                    optimizer.step()

            #perform a prediction on the validation data  
            validation_accuracy_list.append(getAccuracyFromLoader(validation_loader, model))
            training_loss_list.append(loss.data.item())
            
            bar.update(epoch+1)
            
            # early stopping
            early_stopping(loss.data, epoch, model)

            epochs = epochs + 1
            if early_stopping.early_stop:
                print("Early stopping")
                print("total number of epochs: ", epoch)
                break
        
        # Register time
        end = time.time()
        time_elapsed = end - start
        
        # load the last checkpoint with the best model
        model.load_state_dict(early_stopping.getBestModel())
        
        # save information
        if savedModelName is not None:
            # save model
            torch.save(model.state_dict(), os.path.join(savedModelName, CheckpointNameFinal))
            # save results
            with open(os.path.join(savedModelName, accuracyFileName), 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows([validation_accuracy_list])
            with open(os.path.join(savedModelName, lossFileName), 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows([training_loss_list])
            with open(os.path.join(savedModelName, timeFileName), 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow([time_elapsed])
            with open(os.path.join(savedModelName, epochsFileName), 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerow([epochs])
    
    return training_loss_list, validation_accuracy_list, epochs, time_elapsed

def loadModel(model, savedModelName):
    model.load_state_dict(torch.load(os.path.join(savedModelName, CheckpointNameFinal))) 
    model.eval()
    validation_accuracy_list = []
    training_loss_list = []
    time_elapsed = 0
    epochs = 0
    with open(os.path.join(savedModelName, accuracyFileName), newline='') as f:
        reader = csv.reader(f)
        validation_accuracy_list = [float(i) for i in next(reader)] 
    with open(os.path.join(savedModelName, lossFileName), newline='') as f:
        reader = csv.reader(f)
        training_loss_list = [float(i) for i in next(reader)] 
    with open(os.path.join(savedModelName, timeFileName), newline='') as f:
        reader = csv.reader(f)
        time_elapsed = float(next(reader)[0])
    with open(os.path.join(savedModelName, lossFileName), newline='') as f:
        reader = csv.reader(f)
        epochs = float(next(reader)[0])
    return training_loss_list, validation_accuracy_list, epochs, time_elapsed


def getAccuracyFromLoader(loader, model):
    correct=0
    N_test=0
    model.eval()
    for batch in loader:
        with torch.set_grad_enabled(False):
            z = applyModel(batch["image"], model)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == batch["class"]).sum().item()
            N_test = N_test + len(batch["image"])
    return correct / N_test

def getCrossEntropyFromLoader(loader, model):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0, dtype=torch.float)
    lbllist=torch.zeros(0, dtype=torch.long)

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            inputs = batch["image"]
            classes = batch["class"]
            outputs = applyModel(inputs, model)
            _, preds = torch.max(outputs, 1)
            predlist=torch.cat([predlist,outputs], 0)
            lbllist=torch.cat([lbllist,classes], 0)    

    criterion = nn.CrossEntropyLoss()
    return criterion(predlist, lbllist).item()

def getLoaderPredictions(loader, model):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0)
    lbllist=torch.zeros(0)

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            inputs = batch["image"]
            classes = batch["class"]
            outputs = applyModel(inputs, model)
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            predlist=torch.cat([predlist,preds.float().view(-1)])
            lbllist=torch.cat([lbllist,classes.float().view(-1)])
            
    return predlist, lbllist

def applyModel(batch, model):
    if torch.cuda.is_available():
        model_dist = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        outputs = model_dist(batch)
    else:
        outputs = model(batch)
    return outputs