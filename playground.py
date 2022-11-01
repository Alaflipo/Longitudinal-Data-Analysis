import numpy as np 
import pandas as pd 

from src.statistics import Anova

def main(): 
    schooldata = pd.read_csv('data/schooldata.csv')
    schooldata['ARITH_DIF'] = schooldata['POST_ARITH'] - schooldata['PRE_ARITH']
    schooldata['LANG_DIF'] = schooldata['POST_LANG'] - schooldata['PRE_LANG']

    anova = Anova(schooldata)
    anova.set_groups('CLASS', 'ARITH_DIF')
    anova_table = anova.get_anova_table()
    covariance_table = anova.get_covariance_table()
    ICC_table = anova.get_ICC_table()
    print("ANOVA table:")
    print(anova_table)
    print("\n")
    print("Covariance table:")
    print(covariance_table)
    print("\n")
    print("ICC table:")
    print(ICC_table)
    print("\n")

    print(anova.mean_gls)
    anova.plot_residuals()

if __name__ == '__main__': 
    main()