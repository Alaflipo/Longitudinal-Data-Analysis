import numpy as np 
import pandas as pd 

from src.statistics import Stats

def main(): 
    schooldata = pd.read_csv('data/schooldata.csv')
    stats_object = Stats(schooldata)
    stats_object.set_groups('SCHOOL', 'PRE_ARITH')
    print(stats_object.overall_average())

if __name__ == '__main__': 
    main()