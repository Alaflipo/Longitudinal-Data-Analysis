'''
This class holds the Oneway ANOVA table 
@author Morris Boers
@date 2/11/2022 
'''

import pandas as pd
import numpy as np 

class TwowayAnova(): 

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.data_original = data 
        self.n_original = len(data)
        self.c = len(data.iloc[0])

        self.target_name = '' # the target 
        self.fixed_variable_name= '' # higher in hierachy
        self.random_variable_name = '' # nested inside type

        self.groups = {}
        self.group_sizes = []

        self.I = 0 # number of fixed levels (fixed effect)
        self.J = 0 # number of random levels (random effect)
        self.K = 0 # total amount of data points 

        self.df_t = 0 # degrees of freedom for the fixed effect
        self.df_c = 0 # degrees of freedom for the random effect (nested)
        self.df_r = 0 # degrees of freedom for the residual 
        
        self.identified_mean = 0 
        self.identified_level = ''

        # differences with the identified mean 
        self.alphas = []
    
    def set_groups(self, target, fixed, random):
        # drop all missing data 
        self.data = self.data_original.dropna(subset=[target])
        self.K = len(self.data)

        # set target, fixed and random names 
        self.target_name = target 
        self.fixed_variable_name = fixed 
        self.random_variable_name = random 

        # sort data based on first fixed and then random 
        self.data.sort_values(by=[self.fixed_variable_name, self.random_variable_name])

