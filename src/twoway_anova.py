'''
This class holds the Oneway ANOVA table 
@author Morris Boers
@date 2/11/2022 
'''

import pandas as pd
import numpy as np 
import src.utility_functions as uf

class TwowayAnova(): 

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_original = data 
        self.n_original = len(data)
        self.c = len(data.iloc[0])

        self.target_name = '' # the target 
        self.fixed_variable_name= '' # higher in hierachy
        self.random_variable_name = '' # nested inside type

        # list all levels 
        self.fixed_levels = []
        self.random_levels = []

        self.groups = {}
        self.group_sizes_fixed = []
        self.group_sizes_random = []

        self.I = 0 # number of fixed levels (fixed effect)
        self.J = 0 # number of random levels (random effect)
        self.K = 0 # total amount of data points 

        self.df_t = 0 # degrees of freedom for the fixed effect
        self.df_c = 0 # degrees of freedom for the random effect (nested)
        self.df_r = 0 # degrees of freedom for the residual 
        

        # overall mean of all data 
        self.overall_mean = 0 

        # chosen fixed effect mean 
        self.identified_mean = 0 
        self.identified_level = ''

        # differences with the identified mean for each level in the fixed effect
        self.alphas = []
    
    def set_groups(self, target, fixed, random):
        # drop all missing data 
        self.data = self.data_original.dropna(subset=[target, fixed, random])
        self.data.reset_index(inplace=True)
        self.K = len(self.data)

        # set target, fixed and random names 
        self.target_name = target 
        self.fixed_variable_name = fixed 
        self.random_variable_name = random 

        # sort data based on first fixed and then random 
        self.data.sort_values(by=[self.fixed_variable_name, self.random_variable_name])

        # set list of unique levels in each factor 
        self.fixed_levels = self.data[self.fixed_variable_name].unique()
        self.random_levels = self.data[self.random_variable_name].unique()

        # set group sizes
        self.I = len(self.fixed_levels)
        self.J = len(self.random_levels)
        self.group_sizes_fixed = [0] * self.I 
        self.group_sizes_random = [0] * self.J 

        # build nested data structure for fixed - random nesting 
        # also calculate mean for each level in random and fixed 
        for i in range(self.I): 
            values_fixed = [np.array([])]
            self.groups[self.fixed_levels[i]] = {}
            for j in range(self.J): 
                indexes = self.data.index[self.data[self.random_variable_name] == self.random_levels[j]]
                if (self.data.iloc[indexes[0]][self.fixed_variable_name] == self.fixed_levels[i]): 
                    values = self.data.loc[indexes][target].to_numpy()
                    mean = uf.mean(values)
                    self.groups[self.fixed_levels[i]][self.random_levels[j]] = {
                        'values': values, 
                        'size': len(values), 
                        'mean': mean
                    }
                    values_fixed = np.concatenate((values_fixed, values), axis=None)
            mean_fixed = uf.mean(values_fixed)
            self.groups[self.fixed_levels[i]]['mean'] = mean_fixed

        # calculate overall mean 
        self.overall_mean = self.get_overall_mean() 

    def get_overall_mean(self): 
        return uf.mean(self.data[self.target_name].to_numpy())

    



