from math import sqrt
import numpy as np 
import pandas as pd 

from src.statistics import Anova

def main(): 
    schooldata = pd.read_csv('data/balanced_data.csv')
    schooldata['ARITH_DIF'] = schooldata['POST_ARITH'] - schooldata['PRE_ARITH']
    schooldata['LANG_DIF'] = schooldata['POST_LANG'] - schooldata['PRE_LANG']

    anova = Anova(schooldata)
    anova.set_groups('CLASS', 'ARITH_DIF')

    anova_table = anova.get_anova_table()
    covariance_table = anova.get_covariance_table()
    ICC_table = anova.get_ICC_table()

    # print ANOVA, covariance and ICC table 
    print("ANOVA table:")
    print(anova_table)
    print("\n")
    print("Covariance table:")
    print(covariance_table)
    print("\n")
    print("ICC table:")
    print(ICC_table)
    print("\n")

    # plot conditional and marginal residuals 
    # anova.plot_residuals()

    print(anova.ml_mean)
    print(anova.ml_sigma_e_squared)
    print(anova.ml_sigma_g_squared)
    print(sqrt(anova.ml_standard_error_mean))
    print(anova.ml_standard_error_sigma_g)
    print(anova.ml_standard_error_sigma_e)
    print(anova.m)

if __name__ == '__main__': 
    main()