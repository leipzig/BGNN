import os
from torch.utils.data import Dataset
import re
import warnings
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from skimage import io
import time
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle

import progressbar

shuffle_dataset = True

testLoaderFileName = "testLoader.pkl"

def loadTestLoader(experimentName):
    with open(experimentName+"/"+testLoaderFileName, 'rb') as f:
        test_loader = pickle.load(f)
    return test_loader

# Creates the train/val/test dataloaders out of the dataset 
def getLoadersFromDataset(dataset, params, experimentName):
    speciesList = dataset.getSpeciesList()
    train_indices = []
    val_indices = []
    test_indices = []
    
    training_count = params["training_count"]
    validation_count = params["validation_count"]
    batchSize = params["batchSize"]
    
    # for each species, get indices for different sets
    for species in speciesList:
        dataset_size = dataset.getNumberOfImagesForSpecies(species)
        indices = dataset.getSpeciesIndices(species)
#         print("indices", indices)
        split_train = int(training_count)
        split_validation = int(training_count + validation_count)
        if shuffle_dataset :
            np.random.seed(int(time.time()))
            np.random.shuffle(indices)
        
        # aggregate indices
        sub_train_indices, sub_val_indices, sub_test_indices = indices[:split_train], indices[split_train:split_validation], indices[split_validation:]
        train_indices = train_indices + sub_train_indices
        test_indices = test_indices + sub_test_indices
        val_indices = val_indices + sub_val_indices

    # create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # create data loaders.
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batchSize)
    validation_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batchSize)
    test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=batchSize)
    
    # pickle the test_loader to make sure  we always test on unseen data
    fullFileName = experimentName+"/"+testLoaderFileName
    if not os.path.exists(experimentName):
        os.makedirs(experimentName)
    with open(fullFileName, 'wb') as f:
        pickle.dump(test_loader, f)

    return train_loader, validation_loader, test_loader


from ZCA_whitening import ZCA
        
class FishDataset(Dataset):
    def __init__(self, params, verbose=False):
        self.samples = [] # The list of all samples
        self.speciesDictionary = {} # used to get information about each species.
        self.speciesIndexer = [] # used to replace name with a simple number.
        self.data_root = params["image_path"]
        self.imageDimension = params["imageDimension"]
        self.n_channels = params["n_channels"]
        self.useZCAWhitening = params["useZCAWhitening"]
        self.useZCAWhitening = params["useZCAWhitening"]

        index = 0
        # for each file, create a data object
        for fileName in os.listdir(self.data_root):
            match = re.match(r"([a-z,\s]+)(\s)([0-9]+)", fileName, re.I)
            if match:
                (species, dummy, num) = match.groups()
                sampleInfo = {
                    'species': species,
                    'number': num,
                    'index': index,
                    'fileName': fileName,
                    'image': io.imread(os.path.join(self.data_root, fileName))
                }
                self.samples.append(sampleInfo)
                
                # create a dictionary of species
                if (species not in self.speciesDictionary):
                    self.speciesDictionary[species] = [sampleInfo]
                    self.speciesIndexer.append(species)
                else:
                    self.speciesDictionary[species].append(sampleInfo)
                index = index + 1
            else:
                warnings.warn("Could not find a match for " + sampleInfo['fileName'])
        
        transformsList = [transforms.ToPILImage(),
              transforms.Lambda(self.MakeSquared),
              transforms.ToTensor()]
        if self.n_channels == 1:
            transformsList.insert(1, transforms.Grayscale())
                
        # Calculate whitening matrix
        if self.useZCAWhitening:
            self.transforms = transforms.Compose(transformsList)
            zca = ZCA(self)
            transformsList = transformsList + zca.getTransform()
        
        self.transforms = transforms.Compose(transformsList)
        
        if verbose:
            print("Number of images = ", len(self.samples))
            for i in range(len(self.speciesIndexer)):
                species = self.getSpeciesOfIndex(i)
                numOfImages = len(self.speciesDictionary[species])
                print(i, species, " has ", numOfImages, " images")

    
    # Makes the image squared while still preserving the aspect ratio
    def MakeSquared(self, img):
        img_H = img.size[0]
        img_W = img.size[1]
        
        # Resize
        smaller_dimension = 0 if img_H < img_W else 1
        larger_dimension = 1 if img_H < img_W else 0
        new_smaller_dimension = int(self.imageDimension * img.size[smaller_dimension] / img.size[larger_dimension])
        if smaller_dimension == 1:
            img = transforms.functional.resize(img, (new_smaller_dimension, self.imageDimension))
        else:
            img = transforms.functional.resize(img, (self.imageDimension, new_smaller_dimension))

        # pad
        diff = self.imageDimension - new_smaller_dimension
        pad_1 = int(diff/2)
        pad_2 = diff - pad_1
        fill = 255
        if self.n_channels != 1:
            fill = (255, 255, 255)
        if smaller_dimension == 0:
            img = transforms.functional.pad(img, (pad_1, 0, pad_2, 0), padding_mode='constant', fill = fill)
        else:
            img = transforms.functional.pad(img, (0, pad_1, 0, pad_2), padding_mode='constant', fill = fill)

        return img

    def __len__(self):
        return len(self.samples)
    
    # The list of species names
    def getSpeciesList(self):
        return self.speciesIndexer
    
    def getNumberOfImagesForSpecies(self, species):
        return len(self.speciesDictionary[species])
    
    # returns the indices of a species in self.samples
    def getSpeciesIndices(self, species):
        indices = list(map(lambda x: x['index'], self.speciesDictionary[species]))
        return indices
    
    def getSpeciesOfIndex(self, index):
        return self.speciesIndexer[index]

    def __getitem__(self, idx):       
        img_species = self.samples[idx]['species']
        image = self.samples[idx]['image']
        image = self.transforms(image)

        if torch.cuda.is_available():
            image = image.cuda()

        return {'image': image, 'class': self.speciesIndexer.index(img_species)} 