import pandas as pd
from IPython.display import display, HTML
from configParser import getModelName
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from confusion_matrix_plotter import plot_confusion_matrix2, generate_classification_report
import statistics
# import qgrid
import matplotlib.pyplot as plt

from pivottablejs import pivot_ui

aggregateStatFileName = "agg_experiments.csv"
rawStatFileName = "raw_experiments.csv"


class Species_Genus_Statistics:
    def __init__(self, cm, dataset):
        self.dataset = dataset
        self.cm = cm
    
    def get_statistics(self, species_index):
        true_positives = self.cm[species_index, species_index]
        species_name = self.dataset.getSpeciesOfIndex(species_index)
        species_names = self.dataset.getSpeciesWithinGenus(self.dataset.getGenusFromSpecies(species_name))
        species_indexes = list(map(lambda x: self.dataset.getSpeciesList().index(x), species_names))

        within_genus_FP = np.sum(self.cm[species_indexes, species_index]) - true_positives
        out_of_genus_FP = np.sum(self.cm[:, species_index]) - true_positives - within_genus_FP
        within_genus_FN = np.sum(self.cm[species_index, species_indexes]) - true_positives
        out_of_genus_FN = np.sum(self.cm[species_index, :]) - true_positives - within_genus_FN
        return {
            "TP": true_positives,
            "FP_within_genus": within_genus_FP,
            "FP_out_of_genus": out_of_genus_FP,
            "FN_within_genus": within_genus_FN,
            "FN_out_of_genus": out_of_genus_FN,
        }

    def get_precision_recall(self, species_index):
        species_statistics = self.get_statistics(species_index)
        
        TP = species_statistics["TP"]
        FP_within_genus = species_statistics["FP_within_genus"]
        FP_out_of_genus = species_statistics["FP_out_of_genus"]
        FN_within_genus = species_statistics["FN_within_genus"]
        FN_out_of_genus = species_statistics["FN_out_of_genus"]
        
        Precision_within_genus = TP/(TP + FP_within_genus) if (TP + FP_within_genus) != 0 else 0
        Precision_out_of_genus = TP/(TP + FP_out_of_genus) if (TP + FP_out_of_genus) != 0 else 0
        Recall_within_genus = TP/(TP + FN_within_genus) if (TP + FN_within_genus) != 0 else 0
        Recall_out_of_genus = TP/(TP + FN_out_of_genus) if (TP + FN_out_of_genus) != 0 else 0
        Precision = TP/(TP + FP_within_genus + FP_out_of_genus) if (TP + FP_within_genus + FP_out_of_genus) != 0 else 0
        Recall = TP/(TP + FN_within_genus + FN_out_of_genus) if (TP + FN_within_genus + FN_out_of_genus) != 0 else 0
        
        return {
            "Precision_within_genus": Precision_within_genus,
            "Precision_out_of_genus": Precision_out_of_genus,
            "Recall_within_genus": Recall_within_genus,
            "Recall_out_of_genus": Recall_out_of_genus,
            "Precision": Precision,
            "Recall": Recall,
        }
    
    def get_F1Scores(self, species_index):
        precision_recall_stats = self.get_precision_recall(species_index)
        
        within_genus_stats = [precision_recall_stats["Precision_within_genus"], precision_recall_stats["Recall_within_genus"]]
        f1_macro_within_genus = statistics.harmonic_mean(within_genus_stats)
        
        out_of_genus_stats = [precision_recall_stats["Precision_out_of_genus"], precision_recall_stats["Recall_out_of_genus"]]
        f1_macro_out_of_genus = statistics.harmonic_mean(out_of_genus_stats)
        
        overall_stats = [precision_recall_stats["Precision"], precision_recall_stats["Recall"]]
        f1_macro = statistics.harmonic_mean(overall_stats)
        
        return {
            "f1_macro_within_genus": f1_macro_within_genus,
            "f1_macro_out_of_genus": f1_macro_out_of_genus,
            "f1_macro": f1_macro,
        }

class Genus_Statistics:
    def __init__(self, cm, dataset):
        self.dataset = dataset
        self.cm = cm
    
    def get_statistics(self, genus_index):
        true_positives = self.cm[genus_index, genus_index]

        FP = np.sum(self.cm[:, genus_index]) - true_positives
        FN = np.sum(self.cm[genus_index, :]) - true_positives
        return {
            "TP": true_positives,
            "FP": FP,
            "FN": FN,
        }

    def get_precision_recall(self, genus_index):
        species_statistics = self.get_statistics(genus_index)
        
        TP = species_statistics["TP"]
        FP = species_statistics["FP"]
        FN = species_statistics["FN"]
        
        Precision = TP/(TP + FP) if (TP + FP) != 0 else 0
        Recall = TP/(TP + FN) if (TP + FN) != 0 else 0
        
        return {
            "Precision": Precision,
            "Recall": Recall,
        }
    
    def get_F1Scores(self, genus_index):
        precision_recall_stats = self.get_precision_recall(genus_index)
        
        overall_stats = [precision_recall_stats["Precision"], precision_recall_stats["Recall"]]
        f1_macro = statistics.harmonic_mean(overall_stats)
        
        return {
            "f1_macro": f1_macro,
        }


class TrialStatistics:
    def __init__(self, experiment_name, prefix=None):
        self.df = pd.DataFrame()
        self.agg_df = pd.DataFrame()
        
        self.experiment_name = experiment_name
        self.prefix = prefix
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
            file_name = aggregateStatFileName
        else:
            file_name = rawStatFileName
            
        if self.prefix is not None:
            file_name = self.prefix + "_" + file_name
            
        if aggregated:
            if self.agg_df.empty:
                self.aggregateTrials()
            self.agg_df.to_csv(os.path.join(self.experiment_name, file_name))
        else:
            self.df.to_csv(os.path.join(self.experiment_name, file_name))  
        
    def showStatistics(self, aggregated=True):
        df = self.df.copy()
        if aggregated:
            if self.agg_df.empty:
                self.aggregateTrials()
            df = self.agg_df.copy()
            
        name = "Aggregated statistics" if aggregated else "Raw statistics"
        name_html = name+'.html'
        print(name)
#         df.columns = [' '.join(col).strip() for col in df.columns.values] # work around:https://github.com/quantopian/qgrid/issues/18#issuecomment-149321165
#         return qgrid.show_grid(df, show_toolbar=True)
        display(HTML(df.to_html()))
        pivot_ui(df,outfile_path=os.path.join(self.experiment_name, name_html))
#         display(HTML(self.experiment_name+"/"+name_html))
            
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
        
    def prepareConfusionMatrix(self, trial_params):
        if not self.agg_confusionMatrices:
            self.aggregateTrialConfusionMatrices()
    
    def getTrialConfusionMatrix(self, trial_params):
        self.prepareConfusionMatrix(trial_params)
        
        trial_params_copy = self.preProcessParameters(trial_params)
        trial_hash = hash(frozenset(trial_params_copy.items()))
        return self.agg_confusionMatrices[trial_hash]
    
    def printTrialConfusionMatrix(self, trial_params, speciesList, printOutput=False):
        aggregatePath = os.path.join(self.experiment_name, getModelName(trial_params))
        if not os.path.exists(aggregatePath):
            os.makedirs(aggregatePath)
            
        return plot_confusion_matrix2(self.getTrialConfusionMatrix(trial_params),
                                  speciesList,
                                  aggregatePath,
                                  printOutput)
    
    def printF1table(self, trial_params, dataset):
        cm = self.getTrialConfusionMatrix(trial_params)

        if self.prefix == "genus":
            columns = ['genus', 'F1']
        else:
            columns = ['species', 'genus', 'F1']
#             if trial_params['useHeirarchy']:
            columns = columns + ['F1_within_genus', 'F1_out_of_genus']
                
        df = pd.DataFrame(columns=columns)

        if self.prefix == "genus":
            stats = Genus_Statistics(cm, dataset)
            for genus_name in dataset.getGenusList():
                genus_index = dataset.getGenusList().index(genus_name)
                genus_stats = stats.get_F1Scores(genus_index)
                df.loc[genus_index] = [" ".join([str(genus_index), genus_name]),
                                   genus_stats["f1_macro"]]
        else:
            stats = Species_Genus_Statistics(cm, dataset)
            for species in range(len(dataset.getSpeciesList())):
                species_stats = stats.get_F1Scores(species)
                species_name = dataset.getSpeciesOfIndex(species)
                genus_name = dataset.getGenusFromSpecies(species_name)
                genus_index = dataset.getGenusList().index(genus_name)
                vals = [" ".join([str(species), species_name]),
                                   " ".join([str(genus_index), genus_name]),
                                   species_stats["f1_macro"],]
#                 if trial_params['useHeirarchy']:
                vals = vals + [ species_stats["f1_macro_within_genus"],
                               species_stats["f1_macro_out_of_genus"]]
                df.loc[species] = vals
            
        display(HTML(df.to_html()))
        file_name = "F1_Scores"
        if self.prefix == "genus":
            file_name = file_name + "_genus"
        pivot_ui(df,outfile_path=os.path.join(self.experiment_name, file_name+".html"))
    
    def trialScatter(self, x, y, aggregated=True, aggregatedBy=None, save_plot=True):
        df = self.agg_df
        if not aggregated:
            df = self.df
            
        file_name = ('raw' if not aggregated else 'aggregated') + ((' by ' + aggregatedBy) if aggregatedBy is not None else '')
        plot_name = self.experiment_name + ' - ' + file_name
                 
        
        # get unique values for aggregate by
        uniqueValues=['all']
        if aggregatedBy is not None:
            uniqueValues=df[aggregatedBy].unique()       

        if aggregated:
            x_std = (x, 'std')
            y_std = (y, 'std')
            x = (x, 'mean') 
            y = (y, 'mean') 

        # prepare axis
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(plot_name)
        
                     
        for val in uniqueValues:
            if aggregatedBy:
                x_values = df.loc[df[aggregatedBy] == val][x].values
                y_values = df.loc[df[aggregatedBy] == val][y].values    
            else:
                x_values = df[x].values
                y_values = df[y].values

            im = ax.scatter(x=x_values,
                          y=y_values,
                          label=val)

            if aggregated:
                if aggregatedBy:
                    x_std_values = df.loc[df[aggregatedBy] == val][x_std].values
                    y_std_values = df.loc[df[aggregatedBy] == val][y_std].values    
                else:
                    x_std_values = df[x_std].values
                    y_std_values = df[y_std].values
                ax.errorbar(x_values, y_values, yerr=y_std_values, xerr=x_std_values, fmt='*')
            ax.legend()
            
        if save_plot:
            if not os.path.exists(self.experiment_name):
                os.makedirs(self.experiment_name)
            fig.savefig(os.path.join(self.experiment_name, file_name+".pdf"))

        
# TODO: handling classificationReport = generate_classification_report(lbllist, predlist, numberOfSpecies, experimentName)        
    def preProcessParameters(self, trial_params):
        trial_params_copy = {**trial_params, **{}}
        return trial_params_copy
    
# TODO: any place to optimize memory usage?