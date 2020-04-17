import uuid 
import json
import os
import itertools
import pickle 
from hyperopt import fmin, hp, STATUS_OK, Trials, space_eval, plotting, rand, tpe

configJsonFileName = "params.json"
hyperpConfigJsonFileName = "hyperp_params.json"
hyperpSearchConfigFileName = "hyperp_search_params.pkl"
hyperpConfigSelectedJsonFileName = "hyperp_selected_params.pkl"

def getDatasetName(params):
    training_count = params["training_count"]
    validation_count = params["validation_count"]
    imageDimension = params["imageDimension"]
    n_channels = params["n_channels"]
    augmentation = params["augmentation"]
    normalize = params["normalize"]
    
    datasetName = ('tc%f_vc%f_d%d_c%d_aug%s_n%s') % (training_count, validation_count,
                                                                          imageDimension, n_channels, augmentation, normalize)
   
    return datasetName
    
    
def getModelName(params, trial_id=None):
    batchSize = params["batchSize"]
    n_epochs = params["n_epochs"]
    patience = params["patience"]
    learning_rate = params["learning_rate"]
    useHeirarchy = params["useHeirarchy"]
    # temp
    useRelu = params["useRelu"]
    downsample = params["downsample"]
    downsampleOutput = params["downsampleOutput"]
    takeFromIntermediate = params["takeFromIntermediate"]
    takeFromIntermediateOutput = params["takeFromIntermediateOutput"]
    useAdam = params["useAdam"]
    resnet = params["resnet"]
    fc_layers = params["fc_layers"]
    useSoftmax = params["softmax"]
    
    modelName = ('%s_bs%d_e%d_p%d_lr%f_h%s_relu%s_ds%s_dso%s_tfi%s_tfio%s_adm%s_res%s_fc%s_smax%s') % (getDatasetName(params), batchSize, n_epochs, patience, learning_rate, useHeirarchy, useRelu, downsample, downsampleOutput, takeFromIntermediate, takeFromIntermediateOutput, useAdam, resnet, fc_layers, useSoftmax)
   
    if trial_id is not None:
        modelName = modelName + (("_id%s")%(trial_id))  
    
    return modelName

class ConfigParser:
    def __init__(self, experimentName):
        self.experimentName = experimentName
        self.j = None
        self.hyper_j = None
        self.hyperp_search_pkl = None
        self.hyper_selected_j = None
    
    def read(self):
        fullFileName = os.path.join(self.experimentName,configJsonFileName)
        if os.path.exists(fullFileName):
            f = open(fullFileName,"r")
            self.j = json.loads(f.read())
            f.close()
            return self.j
            
    def write(self, params, fileName=configJsonFileName):
        fullFileName = os.path.join(self.experimentName, fileName)
        if os.path.exists(self.experimentName) and os.path.exists(fullFileName):
            self.experimentName = self.experimentName+"-"+uuid.uuid1().hex
        fullFileName = os.path.join(self.experimentName, fileName)
            
        if not os.path.exists(self.experimentName):
            os.makedirs(self.experimentName)

        if fileName==configJsonFileName or fileName==hyperpConfigJsonFileName:
            j = json.dumps(params)
            f = open(fullFileName,"w")
            f.write(j)
            f.close()            
        else:
            j = params
            with open(fullFileName, 'wb') as f:
                pickle.dump(params, f)
        
        if fileName==configJsonFileName:
            self.j = j
        elif fileName==hyperpConfigJsonFileName:
            self.hyperp_search_pkl = j
        elif fileName==hyperpConfigSelectedJsonFileName:
            self.hyper_selected_j=j
        else:
            self.hyper_j = j
        return fullFileName
 
    def writeHyperp(self, params):
        return self.write(params, hyperpConfigJsonFileName)
    
    def writeHyperpSearch(self, params):
        return self.write(params, hyperpSearchConfigFileName)
    
    def writeHyperpSelected(self, params):
        return self.write(params, hyperpConfigSelectedJsonFileName)

    def getHyperpIter(self):
        fullFileName = os.path.join(self.experimentName,hyperpConfigJsonFileName)
        if os.path.exists(fullFileName):
            if self.j is None:
                self.read()
            f = open(fullFileName,"r")
            hyperp_params = json.loads(f.read())
            f.close()
            
            # get possible experiments
            keys, values = zip(*hyperp_params.items())
            
            experimentList = []
            # create experiment params
            for v in itertools.product(*values):
                experiment_params = dict(zip(keys, v))
                experimentList.append({**self.j, **experiment_params})

            return iter(experimentList)
        
    def getHyperpSelectedIter(self):
        fullFileName = os.path.join(self.experimentName, hyperpConfigSelectedJsonFileName)
        if os.path.exists(fullFileName):
            if self.j is None:
                self.read()
            with open(fullFileName, 'rb') as f:
                hyperp_params = pickle.load(f)
            
            experimentList = []
            # create experiment params
            for expriment in hyperp_params:
                experimentList.append({**self.j, **expriment})

            return iter(experimentList)
    
    def getHyperpSearchObject(self):
        fullFileName = os.path.join(self.experimentName, hyperpSearchConfigFileName)
        if os.path.exists(fullFileName):
            if self.j is None:
                self.read()
                
            with open(fullFileName, 'rb') as f:
                hyperp_search_params = pickle.load(f)
            
            # add missing keys as an hp.choice of one value
            for key in self.j:
                if key not in hyperp_search_params:
                    hyperp_search_params[key] = hp.choice(key, [self.j[key]])

            return hyperp_search_params