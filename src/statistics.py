'''
This is the stats utility method
'''

import math 
import numpy as np
import pandas as pd

class Stats: 

    def __init__(self, data: pd.DataFrame):
        self.data = data 
        self.n = len(data)
        self.c = len(data.iloc[0])
        
        self.target_name = ''
        self.grouping_name = ''

        self.m = 0 
        self.group_sizes = []
        self.groups = []
        self.groups_data = {}

        self.overall_average = 0
        self.total_sum_of_squares = 0
    
    def __set_group_data(self, current_group, previous, group_index): 
        self.groups.append(current_group)
        self.groups_data[str(int(previous))] = {
            'size': self.group_sizes[group_index],
            'values': current_group,
            'group_mean': self.mean(current_group) 
        }


    def set_groups(self, group_column_name, data_column_name):
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

        self.overall_average = self.set_overall_average()
        self.total_sum_of_squares = self.set_total_sum_of_squares()

    def set_overall_average(self): 
        average = 0
        for key in self.groups_data: 
            # if all values are nan, it is not counted for the average
            if not math.isnan(self.groups_data[key]['group_mean']): 
                average += (self.groups_data[key]['size'] / self.n) * self.groups_data[key]['group_mean']

        return average

    # is normally distributed with mean \mu and variance \sigma^2/m
    def mean(self, values): 
        sum_without_missing, missing_count = self.sum(values)
        
        if (len(values) == missing_count): 
            return math.nan
        
        # average of the non-missing values
        average_without_missing = sum_without_missing / (len(values) - missing_count)
        # take the average of the non-missing values
        sum_with_missing = sum_without_missing + (missing_count * average_without_missing)
        return sum_with_missing / len(values)

    
    def sum(self, values): 
        sum = 0
        missing_count = 0
        for value in values: 
            if (not math.isnan(value)):
                sum += value
            else: 
                missing_count += 1

        return sum, missing_count

    def set_total_sum_of_squares(self): 
        tss = 0 
        for i in range(self.n): 
            print(self.data.iloc[i][self.target_name])
            if not math.isnan(self.data.iloc[i][self.target_name]): 
                tss += (self.data.iloc[i][self.target_name] - self.overall_average) ** 2
        return tss 
