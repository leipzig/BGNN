from torch import nn
import torch
import progressbar
from earlystopping import EarlyStopping
import os
from torch.autograd import Variable
import csv

CheckpointName = 'checkpoint.pt'
accuracyFileName = "accuracy.csv"
lossFileName = "loss.csv"

# Build the convolutional Neural Network Class
class CNN(nn.Module):
    
    # Contructor
    def __init__(self, numberOfClasses, imgH, kernels, kernelSize, n_channels=1):
        super(CNN, self).__init__()
        in_ch = n_channels
        i=0
        self.numOfLayers = len(kernels)
        self.numberOfClasses = numberOfClasses
        self.module_list = nn.ModuleList()
        
        out_ch = 0
        max_pool_kernel_size = 2
        
        while i < self.numOfLayers:
            out_ch = kernels[i]
            self.module_list.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernelSize, stride=1, padding=2))
            self.module_list.append(nn.ReLU())
            self.module_list.append(nn.MaxPool2d(kernel_size=max_pool_kernel_size))
            in_ch = out_ch
            i = i + 1
        
        n_size = self._get_conv_output((n_channels, imgH, imgH))
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
      
def trainModel(train_loader, validation_loader, n_epochs, model, savedModelName, patience=20):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    loss_list=[]
    accuracy_list=[]
    
    # early stopping
    early_stopping = EarlyStopping(patience=patience)

    print("Training started...")
    with progressbar.ProgressBar(maxval=n_epochs, redirect_stdout=True) as bar:
        bar.update(0)
        for epoch in range(n_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                z = model(batch["image"])
                loss = criterion(z, batch["class"])
                loss.backward()
                optimizer.step()

            correct=0
            #perform a prediction on the validation  data  
            N_test=0
            for batch in validation_loader:
                z = model(batch["image"])
                _, yhat = torch.max(z.data, 1)
                correct += (yhat == batch["class"]).sum().item()
                N_test = N_test + len(batch["image"])
            accuracy = correct / N_test
            accuracy_list.append(accuracy)
            loss_list.append(loss.data.item())
            
            bar.update(epoch+1)
            
            # early stopping
            early_stopping(loss.data, epoch, model)

            if early_stopping.early_stop:
                print("Early stopping")
                print("total number of epochs: ", epoch)
                break
        
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(CheckpointName))
        
        # save information
        if savedModelName is not None:
            if not os.path.exists(savedModelName):
                os.makedirs(savedModelName)
            # save model
            torch.save(model.state_dict(), savedModelName+"/"+CheckpointName)
            # save results
            with open(savedModelName+"/"+accuracyFileName, 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows([accuracy_list])
            with open(savedModelName+"/"+lossFileName, 'w', newline='') as myfile:
                wr = csv.writer(myfile)
                wr.writerows([loss_list])

        os.remove(CheckpointName)
    
    return loss_list, accuracy_list

def loadModel(model, savedModelName):
    model.load_state_dict(torch.load(savedModelName+"/"+CheckpointName)) 
    model.eval()
    accuracy_list = []
    loss_list = []
    with open(savedModelName+"/"+accuracyFileName, newline='') as f:
        reader = csv.reader(f)
        accuracy_list = [float(i) for i in next(reader)] 
    with open(savedModelName+"/"+lossFileName, newline='') as f:
        reader = csv.reader(f)
        loss_list = [float(i) for i in next(reader)] 
    return loss_list, accuracy_list