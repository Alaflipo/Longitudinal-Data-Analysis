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
    print(anova_table)

    anova.set_groups('CLASS', 'LANG_DIF')
    anova_table = anova.get_anova_table()
    print(anova_table)

if __name__ == '__main__': 
    main()