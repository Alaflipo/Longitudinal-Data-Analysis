import numpy as np 
import pandas as pd 

from src.statistics import Stats

def main(): 
    schooldata = pd.read_csv('data/schooldata.csv')
    schooldata['arith_dif'] = schooldata['POST_ARITH'] - schooldata['PRE_ARITH']

    anova = Stats(schooldata)
    anova.set_groups('SCHOOL', 'arith_dif')

    table = [
        [anova.df_between, anova.between_group_sum_of_squares, anova.mean_squares_between, anova.ems_between, anova.sigma_g_squared, anova.F, anova.p], 
        [anova.df_within, anova.within_group_sum_of_squares, anova.mean_squares_within, anova.ems_within, anova.sigma_e_squared],
    ]

    anova_table = pd.DataFrame(
        table, 
        columns=['df', 'sums_squares', 'mean_sum_squares', 'expected_mean_squares', 'sigma_squared_estimation', 'F', 'p value'], 
        index=['between', 'within']
    )

    print(anova_table)

if __name__ == '__main__': 
    main()