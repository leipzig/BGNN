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

shuffle_dataset = True

def getLoadersFromDataset(dataset, training_count, validation_count, batchSize):
    speciesList = dataset.getSpeciesList()
    train_indices = []
    val_indices = []
    test_indices = []
    
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
#         print(species)
#         print("sub_train_indices", sub_train_indices)
#         print("sub_val_indices", sub_val_indices)
#         print("sub_test_indices", sub_test_indices)
        train_indices = train_indices + sub_train_indices
        test_indices = test_indices + sub_test_indices
        val_indices = val_indices + sub_val_indices
#         print("train_indices", train_indices)
#         print("test_indices", test_indices)
#         print("val_indices", val_indices)

    # create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # create data loaders.
    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batchSize)
    validation_loader = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batchSize)
    test_loader = torch.utils.data.DataLoader(dataset, sampler=test_sampler, batch_size=batchSize)

    return train_loader, validation_loader, test_loader
        
        
        
class FishDataset(Dataset):
    def __init__(self, data_root, img_H, img_W, verbose=False):
        self.samples = [] # The list of all samples
        self.speciesDictionary = {} # used to get information about each species.
        self.speciesIndexer = [] # used to replace name with a simple number.
        self.data_root = data_root
        self.transforms = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((img_H, img_W)),
                                              transforms.ToTensor(),
                                             ])

        index = 0
        # for each file, create a data object
        for fileName in os.listdir(data_root):
            match = re.match(r"([a-z,\s]+)(\s)([0-9]+)", fileName, re.I)
            if match:
                (species, dummy, num) = match.groups()
                sampleInfo = {
                    'species': species,
                    'number': num,
                    'index': index,
                    'fileName': fileName
                }
                self.samples.append(sampleInfo)
                # print(species, "-", num, "-", fileName)
                
                # create a dictionary of species
                if (species not in self.speciesDictionary):
                    self.speciesDictionary[species] = [sampleInfo]
                    self.speciesIndexer.append(species)
                else:
                    self.speciesDictionary[species].append(sampleInfo)
                index = index + 1
            else:
                warnings.warn("Could not find a match for " + sampleInfo['fileName'])
        
        if verbose:
            print("Number of images = ", len(self.samples))
            for i in range(len(self.speciesIndexer)):
                species = self.getSpeciesOfIndex(i)
                numOfImages = len(self.speciesDictionary[species])
                print(i, species, " has ", numOfImages, " images")


    def __len__(self):
        return len(self.samples)
    
    def getSpeciesList(self):
        return self.speciesIndexer
    
    def getNumberOfImagesForSpecies(self, species):
        return len(self.speciesDictionary[species])
    
    # returns the indices of the species in self.samples
    def getSpeciesIndices(self, species):
        indices = list(map(lambda x: x['index'], self.speciesDictionary[species]))
        return indices
    
    def getSpeciesOfIndex(self, index):
        return self.speciesIndexer[index]
    
#     def getSpeciesDictionary(self, index):
#         dict = {}
#         for i in range(len(self.speciesIndexer)):
#             dict[i] = self.speciesIndexer[index]
#         return dict

    def __getitem__(self, idx):
        img_name = self.samples[idx]['fileName']
        img_species = self.samples[idx]['species']
        image = io.imread(os.path.join(self.data_root, img_name))
        
        image = self.transforms(image)
#         print(image.shape)
        if torch.cuda.is_available():
            image = image.cuda()
#         print(image.type())
        return {'image': image, 'class': self.speciesIndexer.index(img_species)} 