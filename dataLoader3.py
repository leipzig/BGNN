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
import pandas as pd
import progressbar
import joblib
from configParser import getDatasetName

shuffle_dataset = True

dataset_fileName = "dataset.pkl"
testLoaderFileName = "testLoader.pkl"
valLoaderFileName = "valLoader.pkl"
trainingLoaderFileName = "trainingLoader.pkl"
testIndexFileName = "testIndex.pkl"
valIndexFileName = "valIndex.pkl"
trainingIndexFileName = "trainingIndex.pkl"

image_subpath = "/images"
species_csv_fileName = "metadata.csv"
cleaned_species_csv_fileName = "cleaned_metadata.csv"
statistic_countPerSpecies="count_per_species.csv"
statistic_countPerFamilyAndGenis="count_per_family_genis.csv"

species_csv_fileName_header = "fileName"
species_csv_scientificName_header = "scientificName"
species_csv_Genus_header = "Genus"
species_csv_Family_header = "Family"
species_csv_usedColumns = [species_csv_fileName_header,
                          species_csv_scientificName_header,
                          species_csv_Genus_header,
                          species_csv_Family_header]

from ZCA_whitening import ZCA
        
class FishDataset(Dataset):
    def __init__(self, params, verbose=False):
        self.samples = [] # The list of all samples
        self.imageIndicesPerSpecies = {} # A hash map for fast retreival
        self.imageDimension = params["imageDimension"]
        self.n_channels = params["n_channels"]
        self.useZCAWhitening = params["useZCAWhitening"]
        self.usePretrained = params["usePretrained"]
        self.data_root, self.suffix  = getParams(params)

        if not os.path.exists(self.data_root+"/"+self.suffix):
            os.makedirs(self.data_root+"/"+self.suffix)
        
        # Create species_csv
        cleaned_species_csv_fileName_withsuffix = cleaned_species_csv_fileName
        cleaned_species_csv_fileName_full_path = self.data_root+"/"+self.suffix + cleaned_species_csv_fileName_withsuffix
        cleaned_species_csv_file_exists = os.path.exists(cleaned_species_csv_fileName_full_path)
        if not cleaned_species_csv_file_exists:
            # Load csv file, remove duplicates and invalid, sort.
            csv_full_path = self.data_root + "/" + species_csv_fileName
            self.species_csv = pd.read_csv(csv_full_path, delimiter='\t', index_col=species_csv_fileName_header, usecols=species_csv_usedColumns)
            self.species_csv = self.species_csv.loc[~self.species_csv.index.duplicated(keep='first')]                         
            self.species_csv = self.species_csv[self.species_csv[species_csv_Genus_header] != '#VALUE!']
            
        else:
            self.species_csv = pd.read_csv(cleaned_species_csv_fileName_full_path, delimiter='\t', index_col=species_csv_fileName_header, usecols=species_csv_usedColumns) 
            
        self.species_csv = self.species_csv.sort_values(by=[species_csv_Family_header, species_csv_Genus_header])
        

        # for each file, create a data object
        img_full_path = self.data_root+image_subpath
        
        print("Loading dataset...")
        
        #get intersection between csv file and list of images
        fileNames1 = os.listdir(img_full_path)
        fileNames2 = self.species_csv.index.values.tolist()
        fileNames = [value for value in fileNames1 if value in fileNames2]
        
        index = 0        
        with progressbar.ProgressBar(maxval=len(fileNames), redirect_stdout=True) as bar:
            bar.update(0)
            
            print("Going through image files")
            FoundFileNames = []
            for fileName in fileNames:
                try:
                    # Find match in csv file
                    matchRow = self.species_csv.loc[fileName]
                    matchSpecies = matchRow[species_csv_scientificName_header]
                    matchFamily = matchRow[species_csv_Family_header]
                    matchGenus = matchRow[species_csv_Genus_header]

                    sampleInfo = {
                        'species': matchSpecies,
                        'family': matchFamily,
                        'genus': matchGenus,
                        'number': fileName,
                        'index': fileName,
                        'fileName': fileName,
                        'image': io.imread(os.path.join(img_full_path, fileName))
                    }
                    self.samples.append(sampleInfo)

                    FoundFileNames.append(fileName)
                except:
                    pass

                index = index + 1
                bar.update(index)

            # clean up the csv file from unfound images
            if not cleaned_species_csv_file_exists:
                self.species_csv = self.species_csv.loc[FoundFileNames]
                self.species_csv.to_csv(cleaned_species_csv_fileName_full_path, sep='\t')

            # generate/save statistics on the dataset
            filesPerSpecies_table = self.species_csv[species_csv_scientificName_header].reset_index().groupby(species_csv_scientificName_header).agg('count').sort_values(by=[species_csv_fileName_header]).rename(columns={species_csv_fileName_header: "count"})
            filesPerSpecies_table.to_csv(self.data_root + "/" + self.suffix + statistic_countPerSpecies)
            filesPerFamilyAndGenis_table = self.species_csv[[species_csv_Family_header, species_csv_Genus_header]].reset_index().groupby([species_csv_Family_header, species_csv_Genus_header]).agg('count').sort_values(by=[species_csv_Family_header, species_csv_Genus_header]).rename(columns={species_csv_fileName_header: "count"})
            filesPerFamilyAndGenis_table.to_csv(self.data_root + "/" + self.suffix + statistic_countPerFamilyAndGenis)

        # Create transfroms
        if self.usePretrained:
            transformsList = self.createPretrainedTransforms()
        else:
            transformsList = self.createTransforms()
        self.transforms = transforms.Compose(transformsList)
    
    def createTransforms(self):
        # Create transforms
        transformsList = [transforms.ToPILImage(),
              transforms.Lambda(self.MakeSquared),
              transforms.ToTensor()]
        if self.n_channels == 1:
            transformsList.insert(1, transforms.Grayscale())
                
        # Calculate whitening matrix
        if self.useZCAWhitening:
            print("Calculating ZCA")
            self.transforms = transforms.Compose(transformsList)
            zca = ZCA(self)
            transformsList = transformsList + zca.getTransform()
            print("Calculating ZCA done")
        
        return transformsList
        
    def createPretrainedTransforms(self):
        transformsList = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]       
        return transformsList
        
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
        return self.species_csv[species_csv_scientificName_header].unique().tolist()
    
    def getNumberOfImagesForSpecies(self, species):
#         return len(self.getSpeciesIndices(species))
        countBySpecies_table = self.species_csv[species_csv_scientificName_header].reset_index().groupby(species_csv_scientificName_header).count()
        selectedCount = countBySpecies_table.loc[species]
        return selectedCount[0].item()
    
    # returns the indices of a species in self.samples
    def getSpeciesIndices(self, species):
        if species not in self.imageIndicesPerSpecies:
            self.imageIndicesPerSpecies[species] = [i for i in range(len(self.samples)) if self.samples[i]['species'] == species]
        a = self.species_csv[species_csv_scientificName_header].reset_index()
        a = a[a[species_csv_scientificName_header]==species].index.tolist() 
        return self.imageIndicesPerSpecies[species]
    
    # Convert index to species name.
    def getSpeciesOfIndex(self, index):
        return self.getSpeciesList()[index]

    def __getitem__(self, idx):       
        img_species = self.samples[idx]['species']
        image = self.samples[idx]['image']
        image = self.transforms(image)
        fileName = self.samples[idx]['fileName']

        if torch.cuda.is_available():
            image = image.cuda()

        return {'image': image, 'class': self.getSpeciesList().index(img_species), 'fileName': fileName} 
    

def writeFile(folder_name, file_name, obj):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    try:
        with open(file_name, 'wb') as f:
            joblib.dump(obj, f) 
            print('file',file_name,'written')
    except:
        print("Couldn't write pickle", file_name)
        pass
        
def readFile(fullFileName):
    try:
        with open(fullFileName, 'rb') as filehandle:
            print('file',fullFileName,'read')
            return joblib.load(filehandle) 
    except:
        print("Couldn't read pickle", fullFileName)
        pass  
    

def getParams(params):
    data_root = params["image_path"]
    suffix = str(params["suffix"])+"/" if ("suffix" in params and params["suffix"] is not None) else ""    
    return data_root, suffix
    
class datasetManager:
    def __init__(self, experimentName, verbose=False):
        self.verbose = verbose
        self.suffix = None
        self.data_root = None
        self.experimentName = experimentName
        self.datasetName = None
        self.reset()
    
    def reset(self):
        self.dataset = None
        self.train_loader = None
        self.validation_loader =  None
        self.test_loader = None
    
    def updateParams(self, params):
        datasetName = getDatasetName(params)
        if datasetName != self.datasetName:
            self.reset()
            self.params = params
            self.data_root, self.suffix = getParams(params)
            self.experiment_folder_name = self.data_root + "/" + self.suffix + self.experimentName + "/"
            self.dataset_folder_name = self.experiment_folder_name +  datasetName + "/"
            self.datasetName = datasetName
        
    def getDataset(self):
        saved_dataset_file = self.dataset_folder_name + dataset_fileName
        if not os.path.exists(saved_dataset_file):
            self.dataset = FishDataset(self.params, self.verbose)
            writeFile(self.dataset_folder_name, saved_dataset_file, self.dataset)
        else:
            self.dataset = readFile(saved_dataset_file)
        return self.dataset

    # Creates the train/val/test dataloaders out of the dataset 
    def getLoaders(self):
        if self.dataset is None:
            self.getDataset()
            
        train_loader = None
        validation_loader = None
        test_loader = None
        loaders = []
        loader_fileNames = [trainingLoaderFileName, valLoaderFileName, testLoaderFileName]

        saved_loader_file = self.dataset_folder_name + testLoaderFileName

        if not os.path.exists(saved_loader_file):
            speciesList = self.dataset.getSpeciesList()
            train_indices = []
            val_indices = []
            test_indices = []

            training_count = self.params["training_count"]        
            validation_count = self.params["validation_count"]
            batchSize = self.params["batchSize"]
            
            index_fileNames = [trainingIndexFileName, valIndexFileName, testIndexFileName]
            saved_index_file = self.experiment_folder_name + testIndexFileName
            loader_indices = []
            if not os.path.exists(saved_index_file):

                # for each species, get indices for different sets
                for species in speciesList:
                    dataset_size = self.dataset.getNumberOfImagesForSpecies(species)

                    # if the count is a ratio instead of absolute value, adjust accordingly
                    if training_count < 1:
                        training_count_forSpecies = round(dataset_size*training_count)
                    else:
                        training_count_forSpecies = training_count
                    if validation_count < 1:
                        validation_count_forSpecies = round(dataset_size*validation_count)
                    else:
                        validation_count_forSpecies = validation_count

                    # Logic to find solitting indices for train/val/test.
                    # If dataset_size is too small, there will be overlap.
                    indices = self.dataset.getSpeciesIndices(species)
                    # training set should at least be one element
                    split_train = training_count_forSpecies
                    if split_train == 0:
                        split_train = 1
                    # validation set should start from after training set. But if not enough elements, there will be overlap.
                    # At least one element
                    split_validation_begin = split_train if split_train < dataset_size else dataset_size - 1
                    split_validation = (training_count_forSpecies + validation_count_forSpecies) 
                    if split_validation > dataset_size:
                        split_validation = dataset_size
                    if split_validation == split_validation_begin:
                        split_validation_begin = split_validation_begin - 1
                    # test set is the remaining but at least one element.
                    split_test = split_validation if split_validation < dataset_size else dataset_size-1

                    if shuffle_dataset :
                        np.random.seed(int(time.time()))
                        np.random.shuffle(indices)

                    # aggregate indices
                    sub_train_indices, sub_val_indices, sub_test_indices = indices[:split_train], indices[split_validation_begin:split_validation], indices[split_test:]
                    train_indices = train_indices + sub_train_indices
                    test_indices = test_indices + sub_test_indices
                    val_indices = val_indices + sub_val_indices
                
                # save indices
                loader_indices = [train_indices, val_indices, test_indices]
                for i, name in enumerate(index_fileNames):
                    fullFileName = self.experiment_folder_name+name
                    writeFile(self.experiment_folder_name, fullFileName, loader_indices[i])
                    
            else:
                # load the pickles
                print("Loading saved indices...")
                for i, name in enumerate(index_fileNames):        
                    indiloader_indicesces.append(readFile(self.experiment_folder_name+name))
                
            
            # create samplers
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            # create data loaders.
            train_loader = torch.utils.data.DataLoader(self.dataset, sampler=train_sampler, batch_size=batchSize)
            validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=valid_sampler, batch_size=batchSize)
            test_loader = torch.utils.data.DataLoader(self.dataset, sampler=test_sampler, batch_size=batchSize)
            loaders = [train_loader, validation_loader, test_loader]

            # pickle the loaders
            for i, name in enumerate(loader_fileNames):
                fullFileName = self.dataset_folder_name+name
                writeFile(self.dataset_folder_name, fullFileName, loaders[i])

        else:
            # load the pickles
            print("Loading saved dataloaders...")
            for i, name in enumerate(loader_fileNames):        
                loaders.append(readFile(self.dataset_folder_name+name))

        return loaders[0], loaders[1], loaders[2]