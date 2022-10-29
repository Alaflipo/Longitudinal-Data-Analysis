'''
This is the stats utility method
'''

import math 
import numpy as np
import pandas as pd

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

        self.overall_mean = 0
        self.total_sum_of_squares = 0
        self.within_group_sum_of_squares = 0
        self.between_group_sum_of_squares = 0
    
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
        self.df_between = self.m - 1
        self.df_within = self.n - self.m

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
        missing_count = 0
        for value in values: 
            if (not math.isnan(value)):
                sum += value
            else: 
                missing_count += 1

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
