import pandas as pd
from IPython.display import display, HTML
from configParser import getModelName
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from confusion_matrix_plotter import plot_confusion_matrix2, generate_classification_report

aggregateStatFileName = "agg_experiments.csv"
rawStatFileName = "raw_experiments.csv"

class TrialStatistics:
    def __init__(self, experiment_name):
        self.df = pd.DataFrame()
        self.agg_df = pd.DataFrame()
        
        self.experiment_name = experiment_name
        self.trial_params_keys = []
        self.trial_results_keys = []
        
        self.confusionMatrices = {}
        self.agg_confusionMatrices = {}

    def addTrial(self, trial_params, trial_results, trial=None):
        # Reset aggregate information
        self.agg_df = pd.DataFrame()
        
        # preprocess
        trial_params_copy = self.preProcessParameters(trial_params)
        row_hash = {'hash': hash(frozenset(trial_params_copy.items()))}
        trial_params_with_hash = {**trial_params_copy, **row_hash}
        row_information = {**trial_params_copy, **trial_results}
        row_information = {**row_information, **row_hash}
        
        # Augment row information
        if trial is not None:
            row_information["trial"] = trial

        # Add row
        self.df = self.df.append(pd.DataFrame(row_information, index=[0]), ignore_index = True)
        
        # populate param and result lists keys
        for key in trial_params_with_hash:
            if key not in self.trial_params_keys:
                self.trial_params_keys.append(key)
        
        for key in trial_results:
            if key not in self.trial_results_keys:
                self.trial_results_keys.append(key)
   
    def aggregateTrials(self):        
        # group by trial params
        groupedBy_df = self.df.groupby(self.trial_params_keys)
        
        # For each result key, calculate mean and std
        for key in self.trial_results_keys:
            groupedBy_df_summaried = groupedBy_df.agg({key:['mean','std']})
            self.agg_df = pd.concat([self.agg_df,groupedBy_df_summaried], axis=1, ignore_index=False)
            
        self.agg_df = self.agg_df.reset_index()
        
    def saveStatistics(self, aggregated=True):
        if aggregated:
            if self.agg_df.empty:
                self.aggregateTrials()
            self.agg_df.to_csv(self.experiment_name+"/"+aggregateStatFileName)
        else:
            self.df.to_csv(self.experiment_name+"/"+rawStatFileName)  
        
    def showStatistics(self, aggregated=True):
        if aggregated:
            print("Aggregated statistics")
            if self.agg_df.empty:
                self.aggregateTrials()
            display(HTML(self.agg_df.to_html()))
        else:
            print("Raw statistics")
            display(HTML(self.df.to_html()))
            
    def getStatistic(self, trial_params, metric, statistic):
        if self.agg_df.empty:
            self.aggregateTrials()
        trial_params_copy = self.preProcessParameters(trial_params)
        row = self.agg_df.loc[self.agg_df['hash'] == hash(frozenset(trial_params_copy.items()))]
        return row[self.trial_results_keys][(metric, statistic)].item()
    
    
    
    
    def addTrialPredictions(self, trial_params, predlist, lbllist, numberOfSpecies):
        self.agg_confusionMatrices = {}
        # Confusion matrix
        conf_mat=confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy(), labels = range(numberOfSpecies))
        trial_params_copy = self.preProcessParameters(trial_params)
        trial_hash = hash(frozenset(trial_params_copy.items()))
        if trial_hash not in self.confusionMatrices:
            self.confusionMatrices[trial_hash] = []
        self.confusionMatrices[trial_hash].append(conf_mat)
        
        
    def aggregateTrialConfusionMatrices(self):
        for hash_key in self.confusionMatrices:
            confusionMatricesForHash = self.confusionMatrices[hash_key]
            self.agg_confusionMatrices[hash_key] = np.mean(confusionMatricesForHash, axis=0) 
        
    def printTrialConfusionMatrix(self, trial_params, speciesList, printOutput=False):
        if not self.agg_confusionMatrices:
            self.aggregateTrialConfusionMatrices()
            
        aggregatePath = self.experiment_name+"/"+getModelName(trial_params)
        if not os.path.exists(aggregatePath):
            os.makedirs(aggregatePath)
        
        trial_params_copy = self.preProcessParameters(trial_params)
        trial_hash = hash(frozenset(trial_params_copy.items()))
        return plot_confusion_matrix2(self.agg_confusionMatrices[trial_hash],
                                  speciesList,
                                  aggregatePath,
                                  printOutput)
        
# TODO: handling classificationReport = generate_classification_report(lbllist, predlist, numberOfSpecies, experimentName)        
    def preProcessParameters(self, trial_params):
        trial_params_copy = {**trial_params, **{}}
        trial_params_copy['kernels'] = str(' '.join([str(elem) for elem in trial_params_copy['kernels']]) )
        return trial_params_copy
    
# TODO: any place to optimize memory usage?