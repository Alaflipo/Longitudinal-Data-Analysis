'''
This is the stats utility method
'''

import math 
import numpy as np
import pandas as pd
import scipy

class Stats: 

    def __init__(self, data: pd.DataFrame):
        self.data_original = data 
        self.n_original = len(data)
        self.c = len(data.iloc[0])
        
        self.target_name = ''
        self.grouping_name = ''

        self.data = data 
        self.n = len(data)

        self.m = 0 
        self.group_sizes = []
        self.groups = []
        self.groups_data = {}

        # Mean squares 
        self.overall_mean = 0
        self.total_sum_of_squares = 0
        self.within_group_sum_of_squares = 0
        self.between_group_sum_of_squares = 0
        self.df_between = 0
        self.df_within = 0
        self.mean_squares_within = 0
        self.mean_squares_between = 0

        # F statistic 
        self.F = 0
        self.p = 0 

        # Parameter estimations 
        self.sigma_e_squared = 0 
        self.sigma_g_squared = 0 
        self.C_n = 0 

        # expected mean squares 
        self.ems_between = 0
        self.ems_within = 0
    
    def __set_group_data(self, current_group, previous, group_index): 
        self.groups.append(current_group)
        self.groups_data[str(int(previous))] = {
            'size': self.group_sizes[group_index],
            'values': current_group,
            'group_mean': self.mean(current_group) 
        }


    def set_groups(self, group_column_name, data_column_name):
        self.data = self.data_original.dropna(subset=[data_column_name])
        self.n = len(self.data)

        # set target variable
        self.target_name = data_column_name

        # sort data based on the grouping variable
        self.grouping_name = group_column_name
        self.data.sort_values(by=[self.grouping_name])
        
        # set empty groups and determine amount of groups
        self.m = self.data[self.grouping_name].nunique()
        self.group_sizes = [0] * self.m
        
        # group index will always be < m
        previous =  self.data.iloc[0][self.grouping_name]
        group_index = 0
        current_group = []
        for i in range(self.n): 
            if not (self.data.iloc[i][self.grouping_name] == previous):
                self.__set_group_data(current_group, previous, group_index)
                group_index += 1
                current_group = []
            
            self.group_sizes[group_index] += 1
            current_group.append(self.data.iloc[i][data_column_name])
            previous = self.data.iloc[i][self.grouping_name]
        
        # for the last group
        self.__set_group_data(current_group, previous, group_index)

        self.m = len(self.group_sizes)
        self.overall_mean = self.set_overall_mean()
        self.total_sum_of_squares = self.set_total_sum_of_squares()
        self.within_group_sum_of_squares = self.set_within_group_sum_of_squares()
        self.between_group_sum_of_squares = self.set_between_group_sum_of_squares()

        # set degrees of freedom 
        self.df_within = self.n - self.m
        self.df_between = self.m - 1
        
        # set mean squares 
        self.mean_squares_within = self.within_group_sum_of_squares / self.df_within
        self.mean_squares_between = self.between_group_sum_of_squares / self.df_between

        # F statistic and p value 
        self.F = self.mean_squares_between / self.mean_squares_within
        self.p = self.calculate_p_values()

        # parameter estimation 
        self.sigma_e_squared = self.mean_squares_within
        self.C_n = self.calculate_constant() 
        self.sigma_g_squared = (self.mean_squares_between - self.mean_squares_within) / self.C_n

        # expected mean squares 
        self.ems_between = self.get_ems_between()
        self.ems_within = self.sigma_e_squared

    def set_overall_mean(self): 
        mean = 0
        for key in self.groups_data: 
            # if all values are nan, it is not counted for the average
            if not math.isnan(self.groups_data[key]['group_mean']): 
                mean += (self.groups_data[key]['size'] / self.n) * self.groups_data[key]['group_mean']

        return mean

    # is normally distributed with mean \mu and variance \sigma^2/m
    def mean(self, values): 
        return self.sum(values) / len(values)

    def sos(self, values, average): 
        sos = 0 
        for i in range(len(values)): 
            sos += (values[i] - average) ** 2
        return sos

    def sum(self, values): 
        sum = 0
        for value in values: 
            sum += value

        return sum

    def set_total_sum_of_squares(self): 
        return self.sos(self.data[self.target_name].to_numpy(), self.overall_mean)

    def set_within_group_sum_of_squares(self): 
        within_sos = 0
        for key in self.groups_data: 
            sos = self.sos(self.groups_data[key]['values'], self.groups_data[key]['group_mean'])
            self.groups_data[key]['group_sos'] = sos 
            within_sos += sos
        return within_sos 
    
    def set_between_group_sum_of_squares(self): 
        between_sos = 0 
        for key in self.groups_data: 
            between_sos += self.groups_data[key]['size'] * (self.groups_data[key]['group_mean'] - self.overall_mean) ** 2
        return between_sos 
    
    def calculate_constant(self): 
        # n_bar = self.mean(self.group_sizes) 
        # difference = (np.array(self.group_sizes) - n_bar) ** 2

        sum = 0 
        for i in range(self.m): 
            sum += (self.group_sizes[i] * self.group_sizes[i]) / self.n 

        return (self.n - sum) / (self.m - 1)

    def get_ems_between(self): 
        x = (np.array(self.group_sizes) / self.n)
        return ((self.n - sum(x)) / (self.m - 1)) * self.sigma_g_squared + self.sigma_e_squared

    def calculate_p_values(self): 
        return 1 - scipy.stats.f.cdf(self.F, self.df_between, self.df_within)

