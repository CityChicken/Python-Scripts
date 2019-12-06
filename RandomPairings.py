# -*- coding: utf-8 -*-
"""
This script is used for monte carlo simulations for determining
expected number of weeks until a duplicate pair is produced, assuming
a random pairing of n people each week, where n is and even integer >=2

Created on Sat Oct 26 12:58:18 2019

@author: Jacob
"""
import numpy as np

def ExpectedWeeks(numpeople):
#This will serve as the main function taking in number of people and returning
#the calculated expected number of weeks until a duplicate pair is produced. 
    simulations = 10**5 #set to 10^5 because 10^7 takes an hour or two
    running_total = 0
    test_weeks_array = []
    for i in range(simulations):
        #sum the outcomes and divide
        test_weeks = SimulateWeeks(numpeople)
        test_weeks_array.append(test_weeks)
        running_total += test_weeks
    
    #look at the individual probabilities for each week
    for i in range(numpeople -1):
        print('P(x='+str(i+1)+') =', test_weeks_array.count(i+1) / simulations)
    Ex = running_total/simulations
    return Ex;

def SimulateWeeks(numpeople):
#Generate permutations and store pairs in array
    store_pairs = []
    start_permute = np.random.permutation(numpeople)
    store_pairs = AddPairs(start_permute, store_pairs)
    curr_permute = np.random.permutation(numpeople)
    count = 1
    #continue generating until a pair in curr_permute exists in store_pairs
    while CheckExistence(curr_permute, store_pairs) == False:
        store_pairs = AddPairs(curr_permute, store_pairs)
        curr_permute = np.random.permutation(numpeople)
        count += 1
        #sanity check, since there are only n-1 completely unique
        if count > numpeople - 1:
            print('impossible scenario!!!')
            break
    return count;
    
def AddPairs(add_array, master_array):
#add unique pairs to master_array, "10" and "01", "23" and "32" etc.
    for i in range(len(add_array)//2):
        master_array.append(str(add_array[2*i])+str(add_array[2*i+1]))
        master_array.append(str(add_array[2*i+1])+str(add_array[2*i]))
    return master_array;
    
def CheckExistence(check_array, master_array):
#check if any of the pairs in the current permutation exist in the master collection
    test = False
    for i in range(len(check_array)//2):
        if np.in1d(str(check_array[2*i])+str(check_array[2*i+1]), master_array)[0] == True:
            test = True
            break
    return test;

if __name__ == '__main__':
    print('Expected Weeks =', ExpectedWeeks(8))
