from torch import nn
import torch
import progressbar
from earlystopping import EarlyStopping
import os
from torch.autograd import Variable


# Build the convolutional Neural Network Class
class CNN(nn.Module):
    
    # Contructor
    def __init__(self, numberOfClasses, imgH, imgW, kernels, kernelSize):
        super(CNN, self).__init__()
        n_channels = 3
        in_ch = n_channels
        i=0
        self.numOfLayers = len(kernels)
        self.numberOfClasses = numberOfClasses 
        
        self.cnn = []
        self.relu = []
        self.maxpool = []
        out_ch = 0
        max_pool_kernel_size = 2
        
        while i < self.numOfLayers:
            out_ch = kernels[i]
            self.cnn.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernelSize, stride=1, padding=2))
            self.relu.append(nn.ReLU())
            self.maxpool.append(nn.MaxPool2d(kernel_size=max_pool_kernel_size))
            in_ch = out_ch
            i = i + 1
        
        n_size = self._get_conv_output((n_channels, imgH, imgW))
        self.fc = nn.Linear(n_size, numberOfClasses)
    
    # Prediction
    def forward(self, x):
        out = self.partial_forward(x)
        out = self.fc(out)
        return out
    
    
    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.partial_forward(input)
        return output_feat.shape[1]

    def partial_forward(self, x):
        out = x
        for i in range(self.numOfLayers):
            out = self.cnn[i](out)
            out = self.relu[i](out)
            out = self.maxpool[i](out)
        out = out.view(out.size(0), -1)
        return out
      
def trainModel(train_loader, validation_loader, n_epochs, model, patience=20):
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
            loss_list.append(loss.data)
            
            bar.update(epoch+1)
            
            # early stopping
            early_stopping(loss.data, epoch, model)

            if early_stopping.early_stop:
                print("Early stopping")
                print("total number of epochs: " + epoch)
                break
        
        # load the last checkpoint with the best model
        fileName = 'checkpoint.pt'
        model.load_state_dict(torch.load(fileName))
        os.remove(fileName)
    
    return loss_list, accuracy_list