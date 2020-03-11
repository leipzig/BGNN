import uuid 
import json
import os
import itertools
import pickle 
from hyperopt import fmin, hp, STATUS_OK, Trials, space_eval, plotting, rand, tpe

configJsonFileName = "params.json"
hyperpConfigJsonFileName = "hyperp_params.json"
hyperpSearchConfigFileName = "hyperp_search_params.pkl"

def getModelName(params, trial_id=None):
    training_count = params["training_count"]
    validation_count = params["validation_count"]
    batchSize = params["batchSize"]
    n_epochs = params["n_epochs"]
    kernelSize = params["kernelSize"]
    kernels = '-'.join(map(str, params["kernels"]))  
    patience = params["patience"]
    imageDimension = params["imageDimension"]
    n_channels = params["n_channels"]
    useZCAWhitening = params["useZCAWhitening"]
    
    modelName = ('tc%d_vc%d_bs%d_e%d_ks%d_k%s_p%d_d%d_c%d_zca%s') % (training_count, validation_count, batchSize,
                                                                          n_epochs, kernelSize, kernels, patience, 
                                                                          imageDimension, n_channels,
                                                                          useZCAWhitening)
   
    if trial_id is not None:
        modelName = modelName + (("_id%s")%(trial_id))  
    
    return modelName

class ConfigParser:
    def __init__(self, experimentName):
        self.experimentName = experimentName
        self.j = None
        self.hyper_j = None
        self.hyperp_search_pkl = None
    
    def read(self):
        fullFileName = self.experimentName+"/"+configJsonFileName
        if os.path.exists(fullFileName):
            f = open(fullFileName,"r")
            self.j = json.loads(f.read())
            f.close()
            return self.j
            
    def write(self, params, fileName=configJsonFileName):
        fullFileName = self.experimentName+"/"+fileName
        if os.path.exists(self.experimentName) and os.path.exists(fullFileName):
            self.experimentName = self.experimentName+"-"+uuid.uuid1().hex
        fullFileName = self.experimentName+"/"+fileName
            
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
        else:
            self.hyper_j = j
        return fullFileName
 
    def writeHyperp(self, params):
        return self.write(params, hyperpConfigJsonFileName)
    
    def writeHyperpSearch(self, params):
        return self.write(params, hyperpSearchConfigFileName)

    def getHyperpIter(self):
        fullFileName = self.experimentName+"/"+hyperpConfigJsonFileName
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
        
    def getHyperpSearchObject(self):
        fullFileName = self.experimentName+"/"+hyperpSearchConfigFileName
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