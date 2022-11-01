'''
This is the stats utility method
'''

import math
from statistics import covariance 
import numpy as np
import pandas as pd
import scipy

class Anova: 

    def __init__(self, data: pd.DataFrame):
        self.data_original = data 
        self.n_original = len(data)
        self.c = len(data.iloc[0])

        self.is_balanced = False
        self.alpha = 0.05
        
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
        self.sigma_e_ci_lower, self.sigma_e_ci_upper = 0, 0
        self.standard_error_within_variance = 0 

        self.sigma_g_squared = 0 
        self.sigma_g_ci_lower, self.sigma_g_ci_upper = 0, 0 
        self.standard_error_between_variance = 0 
        self.between_z_value = 0 

        self.C_n = 0 
        self.mean_gls = 0 # generalised least squares 

        self.mean_ols = 0 # ordinary least squares 
        self.ols_ci_lower = 0 # (confidence intervals)
        self.ols_ci_upper = 0

        # expected mean squares 
        self.ems_between = 0
        self.ems_within = 0

        #ICC 
        self.ICC = 0 
        self.F = 0
        self.lower_ICC, self.upper_ICC = 0, 0
    
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

        if (self.group_sizes.count(self.group_sizes[0]) == len(self.group_sizes)):
            self.is_balanced = True

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
        # within group variance estimation (sigma e) and ci 
        self.sigma_e_squared = self.mean_squares_within
        self.standard_error_within_variance = self.get_standard_error_within_variance() 
        self.sigma_e_ci_lower, self.sigma_e_ci_upper = self.get_confidence_intervals_within_group_variance()

        #constant 
        self.C_n = self.calculate_constant() # constant used for calculation

        # between group variance estimation (sigma g) and ci 
        self.sigma_g_squared = (self.mean_squares_between - self.mean_squares_within) / self.C_n
        self.standard_error_between_variance = self.get_standard_error_between_variance()
        self.sigma_g_ci_lower, self.sigma_g_ci_upper = self.get_confidence_intervals_between_group_variance()

        self.mean_gls = self.get_generalised_least_squares()

        # ordinary least squares and it's confidence intervals 
        self.mean_ols = self.get_ordinary_least_squares()
        self.ols_ci_lower, self.ols_ci_upper = self.get_confidence_intervals_ols()

        # expected mean squares 
        self.ems_between = self.get_ems_between()
        self.ems_within = self.sigma_e_squared

        # Interclass correclation coeficient (ICC)
        self.ICC = self.get_ICC()
        self.lower_ICC, self.upper_ICC = self.get_confidence_interval_ICC()

    '''
    creates an ANOVA table based on the calculated statistics 
    class = grouping_name, model (outcome) = target_name 
    returns an anova table with the degrees of freedom (DOF), sums of squares, 
    means squares, expected mean squares and parameter estimators for the variance
    '''
    def get_anova_table(self): 
        table = [
            [   # between group statistics (and F statistic + p value)
                self.df_between, 
                self.between_group_sum_of_squares, 
                self.mean_squares_between, 
                self.ems_between, 
                self.sigma_g_squared, 
                self.F, 
                self.p
            ], 
            [   # within group statistics 
                self.df_within, 
                self.within_group_sum_of_squares, 
                self.mean_squares_within, 
                self.ems_within, 
                self.sigma_e_squared
            ],
        ]

        anova_table = pd.DataFrame(
            table, 
            columns=['df', 'sums_squares', 'mean_sum_squares', 'expected_mean_squares', 'sigma_squared_estimation', 'F', 'p value'], 
            index=['between', 'within']
        )

        return anova_table
    
    def get_covariance_table(self): 

        table = [
            [   # between group variance estimator 
                self.sigma_g_squared, 
                self.standard_error_between_variance, 
                self.between_z_value, 
                math.nan, 
                self.sigma_g_ci_lower, 
                self.sigma_g_ci_upper

            ], 
            [   # within group variance estimator  
                self.sigma_e_squared,
                self.standard_error_within_variance, 
                math.nan, 
                math.nan, 
                self.sigma_e_ci_lower,
                self.sigma_e_ci_upper

            ],
        ]

        covariance_table = pd.DataFrame(
            table, 
            columns=['estimate', 'standard_error', 'z_value', 'p_value', 'lower', 'upper'], 
            index=['group', 'residual']
        )

        return covariance_table

    def get_ICC_table(self): 
        table = [[self.lower_ICC, self.ICC, self.upper_ICC]]
        ICC_table = pd.DataFrame(table, columns=['lower', 'ICC', 'upper'], index=['ICC'])
        return ICC_table


    '''
    Sums the given set of values 
    @param values: holds a list of numeric values 
    @returns the summation of the given values 
    '''
    def sum(self, values): 
        sum = 0
        for value in values: 
            sum += value

        return sum

    '''
    Returns the mean of a set of given values 
    @variable values: holds a list of numeric values 
    @returns the mean of the values  
    '''
    def mean(self, values): 
        return self.sum(values) / len(values)

    '''
    Set's the overall mean by summing over all group means and multiplying it by a weight equal
    to the partion of the whole population 
    @returns the overall mean 
    '''
    def set_overall_mean(self): 
        mean = 0
        for key in self.groups_data: 
            # if all values are nan, it is not counted for the average
            if not math.isnan(self.groups_data[key]['group_mean']): 
                mean += (self.groups_data[key]['size'] / self.n) * self.groups_data[key]['group_mean']

        return mean

    '''
    Returns the sum of squares of a given set of values 
    @param values holds a list of numeric values 
    @param mean the mean of the values 
    @returns the sum of squares from the values
    '''
    
    def sos(self, values, mean): 
        sos = 0 
        for i in range(len(values)): 
            sos += (values[i] - mean) ** 2
        return sos

    '''
    Returns the total sum of squares using the target column 
    @returns the total sum of squares using the overall mean of all values (grouped)
    '''
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
        sum = 0 
        for i in range(self.m): 
            sum += (self.group_sizes[i] * self.group_sizes[i]) / self.n 

        return (self.n - sum) / (self.m - 1)

    def get_ems_between(self): 
        x = (np.array(self.group_sizes) / self.n)
        return ((self.n - sum(x)) / (self.m - 1)) * self.sigma_g_squared + self.sigma_e_squared

    def calculate_p_values(self): 
        # based on alpha value of 0.05
        return 1 - scipy.stats.f.cdf(self.F, self.df_between, self.df_within)
    
    '''
    Calculates the ordinary least squares estimator for the mean 
    @returns the mean OLS estimator 
    '''
    def get_ordinary_least_squares(self): 
        sum = 0
        for i in self.groups_data: 
            for value in self.groups_data[i]['values']: 
                sum += value / self.n
        return sum 

    '''
    Calculates the generalised least squares estimator for the mean (depends on sigma_G and sigma_E)
    @returns the mean GLS estimator 
    '''
    def get_generalised_least_squares(self): 
        numerator = 0 
        denominator = 0 
        for i in self.groups_data: 
            x = (self.sigma_g_squared + self.sigma_e_squared / self.groups_data[i]['size'])
            numerator += self.groups_data[i]['group_mean'] / x
            denominator += 1 / x
        return numerator / denominator
    
    def get_standard_error_between_variance(self): 
        # estimated standard error
        standard_error = (2 * self.mean_squares_between ** 2)/ ((self.group_sizes[0] ** 2) * (self.m - 1))
        standard_error += (2 * self.mean_squares_within ** 2)/ ((self.m * self.group_sizes[0] ** 2) * (self.group_sizes[0] - 1))
        return math.sqrt(standard_error)
    
    def get_standard_error_within_variance(self): 
        # estimated standard error
        return 0 

    '''
    Calculates the confidence intervals of the mean. OLS when data is balanced, GLS when data is unbalanced 
        - OLS: is t distributed and can be calculated exactly (no estimation)
        - GLS: 
    @returns lower and upper intervals of either GLS or OLS 
    '''
    def get_confidence_intervals_ols(self): 

        if (self.is_balanced): 
            t_values = scipy.stats.t(df=(self.m - 1)).ppf((self.alpha / 2, 1 - self.alpha / 2))
            # we are only interested in 1 - alpha / 2 t value 
            variance_ci = t_values[1] * math.sqrt(self.between_group_sum_of_squares/self.n)
            upper_ci = self.mean_ols + variance_ci
            lower_ci = self.mean_ols - variance_ci
            return lower_ci, upper_ci
        else: 
            return 0, 0

    '''
    Calculated the confidence intervals for the within variance estimator 
        - balanced: this is chi squared distributed 
    '''
    def get_confidence_intervals_within_group_variance(self): 

        standard_error = self.standard_error_within_variance

        if (self.is_balanced): 
            chi_value_upper = scipy.stats.chi2.ppf(1 - self.alpha / 2, self.df_within)
            chi_value_lower = scipy.stats.chi2.ppf(self.alpha / 2, self.df_within)
            upper_ci = self.df_within * self.mean_squares_within / chi_value_upper
            lower_ci = self.df_within * self.mean_squares_within / chi_value_lower
            return upper_ci, lower_ci
        else: 
            return 0, 0 
    
    '''
    Calculated the confidence intervals for the between variance estimator 
    There is no exact confidence interval for this so we need to estimate 
    Within this method two different methods are used 
    - one based on a asymptotic approach (used in SAS) based on a normal distribution assumption and 
    - chi squared approximation using the saitterthwaite's approach to calculate the degrees of freedom
    '''
    def get_confidence_intervals_between_group_variance(self): 
        if (self.is_balanced): 
            mode = 'assymptotic'

            standard_error = self.standard_error_between_variance

            if (mode == 'assymptotic'): 
                # assymptotic approach where the normal distribution is used (this is what SAS uses)
                z_value = scipy.stats.norm.ppf(1 - self.alpha / 2) # normal distribution
                self.between_z_value = z_value #!needs editing 
                upper_ci = self.sigma_g_squared + standard_error * z_value
                lower_ci = self.sigma_g_squared - standard_error * z_value
                return lower_ci, upper_ci
            elif (mode == 'saitterthwaites'): 
                # to calculate the degrees of freedom we use saitterthwaites to approach this 
                df_g = 2 * ((self.sigma_g_squared / standard_error) ** 2)
                chi_value_upper = scipy.stats.chi2.ppf(1 - self.alpha / 2, df_g)
                chi_value_lower = scipy.stats.chi2.ppf(self.alpha / 2, df_g)
                upper_ci = df_g * self.mean_squares_within / chi_value_upper
                lower_ci = df_g * self.mean_squares_within / chi_value_lower
                return lower_ci, upper_ci
        else: 
            return 0, 0 
    
    def get_ICC(self):
        if (self.is_balanced):  
            F = self.mean_squares_between / self.mean_squares_within
            self.F = F 
            ICC = (F - 1) / (F + self.group_sizes[0] - 1)
            return ICC 
    
    def get_confidence_interval_ICC(self): 
        if (self.is_balanced): 
            F_L = scipy.stats.f.ppf(self.alpha / 2, self.df_between, self.df_within)
            F_U = scipy.stats.f.ppf(1 - self.alpha / 2, self.df_between, self.df_within)
            lower_ci = (self.F / F_U - 1) / (self.F / F_U + self.group_sizes[0] - 1)
            upper_ci = (self.F / F_L - 1) / (self.F / F_L + self.group_sizes[0] - 1)
            return lower_ci, upper_ci 



