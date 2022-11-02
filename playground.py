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
    
    # print ANOVA, covariance, ICC and maximum likelihood (ML) table 
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
    anova.plot_residuals()

    # only when we have balanced data (FOR NOW)
    if (anova.is_balanced): 
        ML_table = anova.get_ml_table()
        REML_table = anova.get_reml_table()
        comparison_table = anova.get_comparison_table()

        print("ML table:")
        print(ML_table)
        print("\n")
        print("REML table:")
        print(REML_table)
        print("\n")
        print("Comparison of anova, ml and reml estimators:")
        print(comparison_table)
        print("\n")

    
    


if __name__ == '__main__': 
    main()