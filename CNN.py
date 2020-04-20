from torch import nn
import torch
import progressbar
from earlystopping import EarlyStopping
import os
from torch.autograd import Variable
import csv
import time
from sklearn.metrics import confusion_matrix
import collections

CheckpointNameFinal = 'finalModel.pt'

accuracyFileName = "validation_accuracy.csv"
lossFileName = "training_loss.csv"
timeFileName = "time.csv"
epochsFileName = "epochs.csv"

import torchvision.models as models

# Create an FC layer with RELU and/or BatchNormalization
def get_fc(num_of_inputs, num_of_outputs, with_relu = False, with_bnorm = False, num_of_layers = 1):
    l = [] 
    
    for i in range(num_of_layers):
        n_out = num_of_inputs if (i+1 != num_of_layers) else num_of_outputs
        l.append(('linear'+str(i), torch.nn.Linear(num_of_inputs, n_out)))
        if with_bnorm:
            l.append(('bnorm'+str(i), torch.nn.BatchNorm1d(n_out)))
        if with_relu:
            l.append(('relu'+str(i), torch.nn.ReLU()))
    d = collections.OrderedDict(l)
    
    seq = torch.nn.Sequential(d)
    return seq

def create_pretrained_model(params):
    resnet = params["resnet"]
    
    if resnet == "18":
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    return model, num_ftrs

def create_model(architecture, params):
    model = None

    if params["useHeirarchy"]:
        model = CNN_heirarchy(architecture, params)
    else:    
        fc_layers = params["fc_layers"]
        useSoftmax = params["softmax"]
        useRelu = params["useRelu"]
        batchNormalize = params["batchNormalize"]
        
        print('using a pretrained resnet model...')
        model, num_ftrs = create_pretrained_model(params)
        fc = get_fc(num_ftrs, architecture["species"], useRelu, batchNormalize, fc_layers)
        if useSoftmax:
            fc = torch.nn.Sequential(fc, torch.nn.Softmax(dim=1))
        model.fc = fc

    return model

# Build a Hierarchical convolutional Neural Network
class CNN_heirarchy(nn.Module):
    
    # Contructor
    def __init__(self, architecture, params):
        in_ch = params["n_channels"]
        n_channels = in_ch
        imageDimension = params["imageDimension"]
        numberOfClasses = architecture["species"]
        numberOfGenus = architecture["genus"]
        useRelu = params["useRelu"]
        downsample = params["downsample"]
        downsampleOutput = params["downsampleOutput"]
        takeFromIntermediate = params["takeFromIntermediate"]
        takeFromIntermediateOutput = params["takeFromIntermediateOutput"]
        takeFromIntermediateOutput = params["takeFromIntermediateOutput"]
        fc_layers = params["fc_layers"]
        self.useSoftmax = params["softmax"]
        batchNormalize = params["batchNormalize"]
        
        super(CNN_heirarchy, self).__init__()
        self.numberOfClasses = numberOfClasses
        self.numberOfGenus = numberOfGenus
        self.module_list = nn.ModuleList()

        # The pretrained model
        self.pretrained_model, num_ftrs = create_pretrained_model(params)
        resnet_subLayers = [self.pretrained_model.conv1,
          self.pretrained_model.bn1,
          self.pretrained_model.relu,
          self.pretrained_model.maxpool,
          self.pretrained_model.layer1,
          self.pretrained_model.layer2,
          self.pretrained_model.layer3,
          self.pretrained_model.layer4,
          self.pretrained_model.avgpool]
        self.resnet_before_fc = torch.nn.Sequential(*resnet_subLayers)
        self.module_list.append(self.resnet_before_fc)
        
        # Down sampling to species FC
        species_inputs = num_ftrs
        self.downsampled_model = None
        if downsample:
            self.downsampled_model = get_fc(num_ftrs, downsampleOutput, True, batchNormalize, num_of_layers=fc_layers)
            self.module_list.append(self.downsampled_model)
            species_inputs = downsampleOutput
            
        # Take from intermediate of genus FC
        genus_inputs = self.numberOfGenus
        intermediate_input = num_ftrs
        self.intermediate_genus_model = None
        if takeFromIntermediate:
            self.intermediate_genus_model = get_fc(num_ftrs, takeFromIntermediateOutput, True, batchNormalize, num_of_layers=fc_layers)
            self.module_list.append(self.intermediate_genus_model)
            genus_inputs = takeFromIntermediateOutput
            intermediate_input = takeFromIntermediateOutput
        
        # Genus    
        self.genus_fc = get_fc(intermediate_input, self.numberOfGenus, useRelu)
        self.module_list.append(self.genus_fc)
        
        # Adjust last layer of resnet
        if self.intermediate_genus_model:
            self.pretrained_model.fc = self.intermediate_genus_model
        else:
            self.pretrained_model.fc = self.genus_fc
        self.module_list.insert(0, self.pretrained_model)
        
        # the fully connect species later
        self.to_species_layer = torch.nn.Linear(genus_inputs + species_inputs,  numberOfClasses)
        self.module_list.append(self.to_species_layer)
        
        if self.useSoftmax:
            self.softmax_layer = torch.nn.Softmax(dim=1)
    
    # Prediction
    def forward(self, x):
        activations = self.activations(x)
        result = {
            "species": activations["species"],
            "genus" : activations["genus"]
        }
        return result


    default_outputs = {
        "species": True,
        "genus" : True
    }
    def activations(self, x, outputs=default_outputs):
        inpt = x
        intermediate_from_resnet = torch.flatten(self.resnet_before_fc(inpt),1)
        
        genus_intermediate = self.pretrained_model(x)
        if outputs["genus"]:
            if self.intermediate_genus_model is not None:
                genus = self.genus_fc(genus_intermediate)
            else:
                genus = genus_intermediate
            if self.useSoftmax:
                genus = self.softmax_layer(genus)
        
        if self.downsampled_model:
            downsampled_features = self.downsampled_model(intermediate_from_resnet)
        else:
            downsampled_features = intermediate_from_resnet

        if outputs["species"]:
            species = self.to_species_layer(torch.cat((downsampled_features, genus_intermediate), 1))
            if self.useSoftmax:
                species = self.softmax_layer(species)

        activations = {
            "input": inpt,
            "intermediate_resent": intermediate_from_resnet,
            "genus_intermediate": genus_intermediate,
            "downsampled_features": downsampled_features,
            "genus": genus if outputs["genus"] else None,
            "species": species if outputs["species"] else None
        }

        return activations

def getModelFile(experimentName):
    return os.path.join(experimentName, CheckpointNameFinal)
    
def trainModel(train_loader, validation_loader, params, model, savedModelName):
    n_epochs = params["n_epochs"]
    patience = params["patience"]
    learning_rate = params["learning_rate"]
    useHeirarchy = params["useHeirarchy"]
    batchSize = params["batchSize"]
    useAdam = params["useAdam"]
    
    if not os.path.exists(savedModelName):
        os.makedirs(savedModelName)
    
    if useAdam:
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    else:
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
                    
                    loss = None
                    if useHeirarchy:
                        loss = criterion(z["genus"], batch["genus"])
                        loss2 = criterion(z["species"], batch["species"])
                        (loss + loss2).backward()
                    else:    
                        loss = criterion(z, batch["species"])
                        loss.backward()
                    optimizer.step()
            model.eval()

            #perform a prediction on the validation data  
            validation_accuracy_list.append(getAccuracyFromLoader(validation_loader, model, params))
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

# loads a saved model along with its results
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
    with open(os.path.join(savedModelName, epochsFileName), newline='') as f:
        reader = csv.reader(f)
        epochs = float(next(reader)[0])
    return training_loss_list, validation_accuracy_list, epochs, time_elapsed


def getAccuracyFromLoader(loader, model, params):
    useHeirarchy = params["useHeirarchy"]
    
    correct=0
    N_test=0
    model.eval()
    for batch in loader:
        with torch.set_grad_enabled(False):
            z = applyModel(batch["image"], model)
            if useHeirarchy:
                z = z["species"]
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == batch["species"]).sum().item()
            N_test = N_test + len(batch["image"])
    return correct / N_test

def getCrossEntropyFromLoader(loader, model, params, label="species"):
    predlist, lbllist = getLoaderPredictions(loader, model, params, label, False) 

    criterion = nn.CrossEntropyLoss()
    return criterion(predlist, lbllist).item()

def getLoaderPredictions(loader, model, params, label="species", flattenConcat = True):
    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0)
    if flattenConcat:
        lbllist=torch.zeros(0)
    else:
        lbllist=torch.zeros(0, dtype=torch.long)
    useHeirarchy = params["useHeirarchy"]

    model.eval()
    with torch.set_grad_enabled(False):
        for batch in loader:
            inputs = batch["image"]
            classes = batch[label]
            outputs = applyModel(inputs, model)
            if useHeirarchy:
                outputs = outputs[label]
            _, preds = torch.max(outputs, 1)

            # Append batch prediction results
            if flattenConcat:
                predlist=torch.cat([predlist,preds.float().view(-1)])
                lbllist=torch.cat([lbllist,classes.float().view(-1)])
            else:
                predlist=torch.cat([predlist,outputs], 0)
                lbllist=torch.cat([lbllist,classes], 0)  
            
    return predlist, lbllist
def applyModel(batch, model):
    if torch.cuda.is_available():
        model_dist = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        outputs = model_dist(batch)
    else:
        outputs = model(batch)
    return outputs