'''
Sums the given set of values 
@param values: holds a list of numeric values 
@returns the summation of the given values 
'''
def sum(values): 
    sum = 0
    for value in values: 
        sum += value

    return sum

'''
Returns the mean of a set of given values 
@variable values: holds a list of numeric values 
@returns the mean of the values  
'''
def mean(values): 
    return sum(values) / len(values)