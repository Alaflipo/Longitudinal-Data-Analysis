'''
This is the stats utility method
'''

import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
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
        self.n_0 = 0 

        self.m = 0 
        self.group_sizes = []
        self.groups = []
        self.groups_data = {}

        # Mean squares 
        self.overall_mean = 0
        self.total_sum_of_squares = 0
        self.df_total = 0 
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

        self.sigma_t_squared = 0 
        self.standard_error_total_variance = 0
        self.sigma_t_ci_lower, self.sigma_t_ci_upper = 0, 0

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

        # maximum likelihood estimation 
        self.mean_ml = 0
        self.ml_sigma_e_squared = 0
        self.ml_sigma_g_squared = 0 

        self.ml_lambda = 0 
        self.ml_standard_error_mean = 0
        self.ml_standard_error_sigma_g = 0 
        self.ml_standard_error_sigma_e = 0 
    
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
        
        self.n_0 = self.group_sizes[0]

        self.m = len(self.group_sizes)
        self.overall_mean = self.set_overall_mean()
        self.total_sum_of_squares = self.set_total_sum_of_squares()
        self.within_group_sum_of_squares = self.set_within_group_sum_of_squares()
        self.between_group_sum_of_squares = self.set_between_group_sum_of_squares()

        # set degrees of freedom 
        self.df_within = self.n - self.m
        self.df_between = self.m - 1
        
        # set mean squares 
        self.mean_squares_within = self.get_mean_squares_within()
        self.mean_squares_between = self.get_mean_squares_between()

        # F statistic and p value 
        self.F = self.calculate_F_value()
        self.p = self.calculate_p_values()

        # parameter estimation 
        self.C_n = self.calculate_constant() # constant used for calculation
        # within group variance estimation (sigma e) and ci 
        self.sigma_e_squared = self.get_sigma_e_squared()
        self.standard_error_within_variance = self.get_standard_error_within_variance() 
        self.sigma_e_ci_lower, self.sigma_e_ci_upper = self.get_confidence_intervals_within_group_variance()

        # between group variance estimation (sigma g) and ci 
        self.sigma_g_squared = self.get_sigma_g_squared()
        self.standard_error_between_variance = self.get_standard_error_between_variance()
        self.sigma_g_ci_lower, self.sigma_g_ci_upper = self.get_confidence_intervals_between_group_variance()

        # total group variance estimation (sigma t) and ci
        self.sigma_t_squared = self.get_sigma_t_squared()
        self.standard_error_total_variance = self.get_standard_error_total_variance()
        self.sigma_t_ci_lower, self.sigma_t_ci_upper = self.get_confidence_intervals_total_variance()

        # ordinary least squares and it's confidence intervals 
        self.mean_ols = self.get_ordinary_least_squares()
        self.ols_ci_lower, self.ols_ci_upper = self.get_confidence_intervals_ols()

        # generalised least squares and it's confidence intervals 
        self.mean_gls = self.get_generalised_least_squares()

        # expected mean squares 
        self.ems_between = self.get_ems_between()
        self.ems_within = self.get_ems_within()

        # Interclass correclation coeficient (ICC)
        self.ICC = self.get_ICC()
        self.lower_ICC, self.upper_ICC = self.get_confidence_interval_ICC()

        # blubs calculation 
        self.set_blubs()

        # set marginal and conditional residuals 
        self.set_residuals()

        # maximum likelihood estimation 
        self.ml_mean = self.get_ml_mean()
        self.ml_sigma_g_squared = self.get_ml_sigma_g_squared()
        self.ml_sigma_e_squared = self.get_ml_sigma_e_squared()

        self.ml_lambda = self.get_ml_lambda()
        self.ml_standard_error_mean = self.get_ml_standard_error_mean()
        self.ml_standard_error_sigma_g = self.get_ml_standard_error_sigma_g()
        self.ml_standard_error_sigma_e = self.get_ml_standard_error_sigma_e()

    #####################################################
    #                   Visualise data                  #
    #####################################################

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
            [   # total group variance estimator  
                self.sigma_t_squared,
                self.standard_error_total_variance, 
                math.nan, 
                math.nan, 
                self.sigma_t_ci_lower,
                self.sigma_t_ci_upper

            ]
        ]

        covariance_table = pd.DataFrame(
            table, 
            columns=['estimate', 'standard_error', 'z_value', 'p_value', 'lower', 'upper'], 
            index=['group', 'residual', 'total']
        )

        return covariance_table

    def get_ICC_table(self): 
        table = [[self.lower_ICC, self.ICC, self.upper_ICC]]
        ICC_table = pd.DataFrame(table, columns=['lower', 'ICC', 'upper'], index=['ICC'])
        return ICC_table
    
    def plot_residuals(self): 
        #conditional residuals 
        figure, axis = plt.subplots(2, 2)

        # blubs vs. conditional residuals
        axis[0, 0].plot(self.data['blubs'], self.data['conditional_residuals'], 'o')

        # histogram of the condtionals residuals with normal line
        axis[0, 1].hist(self.data['conditional_residuals'], edgecolor='black', bins=24)

        # qqplot of the conditional residuals 
        scipy.stats.probplot(self.data['conditional_residuals'], dist='norm', plot=axis[1, 0])
        axis[1, 0].set_title('')
        axis[1, 0].set_xlabel('')
        axis[1, 0].set_ylabel('')
        plt.show()

        # marginal residuals 
        figure, axis = plt.subplots(2, 2)

        # blubs vs. marginal residuals
        x = [self.mean_gls] * self.n
        axis[0, 0].plot(x, self.data['marginal_residuals'], 'o')

        # histogram of the condtionals residuals with normal line
        axis[0, 1].hist(self.data['marginal_residuals'], edgecolor='black', bins=24)

        # qqplot of the conditional residuals 
        scipy.stats.probplot(self.data['marginal_residuals'], dist='norm', plot=axis[1, 0])
        axis[1, 0].set_title('')
        axis[1, 0].set_xlabel('')
        axis[1, 0].set_ylabel('')
        plt.show()

    
    #####################################################
    #                   Calculations                    #
    #####################################################
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
    
    def calculate_p_values(self): 
        # based on alpha value of 0.05
        return 1 - scipy.stats.f.cdf(self.F, self.df_between, self.df_within)
    
    def calculate_F_value(self):
        return self.between_group_sum_of_squares / self.df_between
    
    #####################################################
    #                   Sums of squares                 #
    #####################################################

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
    
    #####################################################
    #                   Mean squares                    #
    #####################################################
    
    def get_mean_squares_within(self): 
        return self.within_group_sum_of_squares / self.df_within
    
    def get_mean_squares_between(self): 
        return self.between_group_sum_of_squares / self.df_between
    
    #####################################################
    #                Parameter Estimation               #
    #####################################################

    def get_sigma_e_squared(self): 
        return self.mean_squares_within
    
    def calculate_constant(self): 
        sum = 0 
        for i in range(self.m): 
            sum += (self.group_sizes[i] * self.group_sizes[i]) / self.n 

        return (self.n - sum) / (self.m - 1)

    def get_sigma_g_squared(self):
        return (self.mean_squares_between - self.mean_squares_within) / self.C_n

    def get_sigma_t_squared(self):
        return self.sigma_g_squared + self.sigma_e_squared
    
    #####################################################
    #               Expected Mean Squares               #
    #####################################################

    def get_ems_between(self): 
        x = (np.array(self.group_sizes) / self.n)
        return ((self.n - sum(x)) / (self.m - 1)) * self.sigma_g_squared + self.sigma_e_squared
    
    def get_ems_within(self): 
        return self.sigma_e_squared

    #####################################################
    #                   STANDARD ERROR                  #
    #####################################################

    def get_standard_error_total_variance(self): 
        if (self.is_balanced): 
            # estimated standard error
            variance = (2 * self.mean_squares_between ** 2) / ((self.n_0 ** 2) * (self.m - 1))
            variance += ((2 * (self.n_0 - 1) ** 2) * (self.mean_squares_within ** 2)) / ((self.n_0 - 1) * self.m * self.n_0 ** 2)
            return math.sqrt(variance)
        else: 
            # estimated standard error
            variance = (2 * self.mean_squares_between ** 2) / ((self.C_n ** 2) * (self.m - 1))
            variance += ((2 * (self.C_n - 1) ** 2) * (self.mean_squares_within ** 2)) / ((self.n - self.m) * self.C_n ** 2)
            return math.sqrt(variance)
    
    def get_standard_error_between_variance(self): 
        # estimated standard error
        standard_error = (2 * self.mean_squares_between ** 2)/ ((self.C_n ** 2) * (self.m - 1))
        standard_error += (2 * self.mean_squares_within ** 2)/ ((self.m * self.C_n ** 2) * (self.C_n - 1))
        return math.sqrt(standard_error)
    
    def get_standard_error_within_variance(self): 
        return math.sqrt(2 * self.mean_squares_within ** 2/self.df_within)
    
    #####################################################
    #               Confidence Intervals                #
    #####################################################

    def get_confidence_intervals_total_variance(self): 
        variance = self.standard_error_total_variance
        df_total = 2 * (self.sigma_t_squared / variance) ** 2

        chi_value_upper = scipy.stats.chi2.ppf(1 - self.alpha / 2, df_total)
        chi_value_lower = scipy.stats.chi2.ppf(self.alpha / 2, df_total)

        ci_lower = df_total * self.sigma_t_squared / chi_value_upper
        ci_upper = df_total * self.sigma_t_squared / chi_value_lower

        return ci_lower, ci_upper
    
    '''
    Calculated the confidence intervals for the within variance estimator 
        - balanced: this is chi squared distributed 
    '''
    def get_confidence_intervals_within_group_variance(self): 

        # We are not using the standard error for this calculation 
        standard_error = self.standard_error_within_variance

        chi_value_upper = scipy.stats.chi2.ppf(1 - self.alpha / 2, self.df_within)
        chi_value_lower = scipy.stats.chi2.ppf(self.alpha / 2, self.df_within)
        upper_ci = self.df_within * self.mean_squares_within / chi_value_upper
        lower_ci = self.df_within * self.mean_squares_within / chi_value_lower
        return upper_ci, lower_ci
    
    '''
    Calculated the confidence intervals for the between variance estimator 
    There is no exact confidence interval for this so we need to estimate 
    Within this method two different methods are used 
    - one based on a asymptotic approach (used in SAS) based on a normal distribution assumption and 
    - chi squared approximation using the saitterthwaite's approach to calculate the degrees of freedom
    '''
    def get_confidence_intervals_between_group_variance(self): 
        mode = 'saitterthwaites'

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
            df_g = 2 * (self.sigma_g_squared / standard_error) ** 2
            chi_value_upper = scipy.stats.chi2.ppf(1 - self.alpha / 2, df_g)
            chi_value_lower = scipy.stats.chi2.ppf(self.alpha / 2, df_g)
            upper_ci = df_g * self.sigma_g_squared / chi_value_lower
            lower_ci = df_g * self.sigma_g_squared / chi_value_upper
            return lower_ci, upper_ci

    '''
    Calculates the confidence intervals of the mean. OLS when data is balanced, GLS when data is unbalanced 
        - OLS: is t distributed and can be calculated exactly (no estimation)
    @returns lower and upper intervals of either GLS or OLS 
    '''
    def get_confidence_intervals_ols(self): 

        if (self.is_balanced): 
            t_values = scipy.stats.t(df=(self.m - 1)).ppf((self.alpha / 2, 1 - self.alpha / 2))
            # we are only interested in 1 - alpha / 2 t value 
            variance_ci = t_values[1] * math.sqrt(self.mean_squares_between/self.n)
            upper_ci = self.mean_ols + variance_ci
            lower_ci = self.mean_ols - variance_ci
            return lower_ci, upper_ci
        else: 
            return 0, 0
    
    def get_confidence_intervals_gls(self): 
        return 0 
    
    def get_confidence_interval_ICC(self): 
        group_constant = 0
        if (self.is_balanced): 
            group_constant = self.n_0
        else: 
            group_constant = self.C_n

        F_L = scipy.stats.f.ppf(self.alpha / 2, self.df_between, self.df_within)
        F_U = scipy.stats.f.ppf(1 - self.alpha / 2, self.df_between, self.df_within)
        lower_ci = (self.F / F_U - 1) / (self.F / F_U + group_constant - 1)
        upper_ci = (self.F / F_L - 1) / (self.F / F_L + group_constant - 1)

        return lower_ci, upper_ci 
    
    #####################################################
    #               Least Squares Estimation            #
    #####################################################
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

    #####################################################
    #          Intraclass Correlation Coeficient        #
    #####################################################
    
    def get_ICC(self):
        if (self.is_balanced):  
            F = self.mean_squares_between / self.mean_squares_within
            self.F = F 
            ICC = (F - 1) / (F + self.n_0 - 1)
            return ICC 
        else: 
            F = self.mean_squares_between / self.mean_squares_within
            self.F = F 
            ICC = (F - 1) / (F + self.C_n - 1)
            return ICC 
    
    #####################################################
    #      Best Linear Unbiased Predictor (BLUP)        #
    #####################################################

    # sets the blubs which take into account that the class effects are random and should be modeled like that 
    def set_blubs(self): 
        all_blubs = []
        for key in self.groups_data:
            blub = self.mean_gls
            blub += (self.sigma_g_squared / (self.sigma_g_squared + self.sigma_e_squared / self.groups_data[key]['size'])) * \
                (self.groups_data[key]['group_mean'] - self.mean_gls)
            self.groups_data[key]['blub'] = blub 
            for value in self.groups_data[key]['values']: 
                all_blubs.append(blub)
        
        self.data['blubs'] = all_blubs
    
    #####################################################
    #                    Residuals                      #
    #####################################################

    # set's both the marginal and conditional residuals 
    # - condtional: subtracts both the fixed effects and random effects using the blubs 
    # - marginal: subtracts only the fixed effects by subtracting the generalised least squares
    # it can use two different standarization modes: 
    # - standardized residuals (student): which devides by its own variance 
    # - persons's residuals: which devides by the variance of the corresponding observation (implemented later)
    def set_residuals(self): 
        standarisation_mode = 'pearson'
        all_marginal_residuals = []
        all_conditional_residuals = []

        for key in self.groups_data:
            conditional_residuals = []
            marginal_residuals = []
            for data_point in self.groups_data[key]['values']:
                standardized = 0 
                if (standarisation_mode == 'pearson'): 
                    standardized = math.sqrt(self.sigma_e_squared + self.sigma_g_squared)
                else: 
                    standardized = math.sqrt(1)

                # conditional residual calculation
                con_resid = (data_point - self.groups_data[key]['blub']) / standardized
                conditional_residuals.append(con_resid)
                all_conditional_residuals.append(con_resid)
                # marginal residual calculation 
                mar_resid = (data_point - self.mean_gls) / standardized
                marginal_residuals.append(mar_resid)
                all_marginal_residuals.append(mar_resid)

            self.groups_data[key]['conditional_resid'] = conditional_residuals
            self.groups_data[key]['marginal_resid'] = marginal_residuals
        
        self.data['marginal_residuals'] = all_marginal_residuals
        self.data['conditional_residuals'] = all_conditional_residuals

    #####################################################
    #        Maximum Likelihood (ML) estimation         #
    #####################################################
    
    def get_ml_mean(self): 
        return self.overall_mean
    
    def get_ml_sigma_g_squared(self): 
        estimator = ((self.m - 1) * self.mean_squares_between/self.m - self.mean_squares_within) / self.n_0
        return max(0, estimator)

    def get_ml_sigma_e_squared(self): 
        if (self.ml_sigma_g_squared > 0): 
            return self.mean_squares_within
        else: 
            return (self.between_group_sum_of_squares + self.within_group_sum_of_squares) / (self.m * self.n_0)

    def get_ml_lambda(self): 
        return self.n_0 * self.ml_sigma_g_squared + self.ml_sigma_e_squared

    def get_ml_standard_error_mean(self): 
        return self.ml_lambda / (self.m * self.n_0)
    
    def get_ml_standard_error_sigma_g(self): 
        first_part = (2 * self.ml_sigma_e_squared ** 2) / (self.m * self.n_0 ** 2)
        second_part = (1/(self.n_0 - 1)) + (self.ml_lambda ** 2 / self.ml_sigma_e_squared ** 2)
        return first_part * second_part
    
    def get_ml_standard_error_sigma_e(self): 
        self.get_ml_ci_mean()
        return (2 * self.ml_sigma_e_squared ** 2) / (self.m * (self.n_0 - 1))

    def get_ml_ci_mean(self): 
        t_values = scipy.stats.t(df=(self.m - 1)).ppf((self.alpha / 2, 1 - self.alpha / 2))
        t_score = ((self.ml_mean) / self.ml_standard_error_mean) * math.sqrt(self.n)
        print(t_score)
        print(t_values)

        
        
    

    
    




