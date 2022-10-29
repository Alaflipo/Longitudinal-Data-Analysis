import numpy as np 
import pandas as pd 

from src.statistics import Stats

def main(): 
    schooldata = pd.read_csv('data/schooldata.csv')
    stats_object = Stats(schooldata)
    stats_object.set_groups('SCHOOL', 'PRE_ARITH')
    print(stats_object.overall_average)
    print(stats_object.total_sum_of_squares)
    print(stats_object.within_group_sum_of_squares)
    print(stats_object.between_group_sum_of_squares)
    print(stats_object.within_group_sum_of_squares + stats_object.between_group_sum_of_squares)

if __name__ == '__main__': 
    main()