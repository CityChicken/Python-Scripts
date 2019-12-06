# -*- coding: utf-8 -*-
"""
This script is used for monte carlo simulations for determining
expected number of weeks until a duplicate set of n/2 pairs is produced, 
assuming a random pairing of n people each week, where n is and even integer >=2

Created on Sat Oct 26 12:58:18 2019

@author: Jacob
"""
import numpy as np
import math as m
def ExpectedWeeks(numpeople):
#This will serve as the main function taking in number of people and returning
#the calculated expected number of weeks until a duplicate set of 4 pairs is produced. 
    simulations = 10**6 #set to 10^5 because 10^6 takes an hour or two
    running_total = 0
    #test_weeks_array = []
    for i in range(simulations):
        test_weeks = SimulateWeeks(numpeople)
        #test_weeks_array.append(test_weeks)
        running_total += test_weeks
    
    #look at the individual probabilities for each week
    #limit = m.factorial(numpeople)/(2**(numpeople/2) * m.factorial(numpeople/2))
    #for i in range(int(limit)):
     #   print('P(x='+str(i+1)+') =', test_weeks_array.count(i+1) / simulations)
    Ex = running_total/simulations
    return Ex;

def SimulateWeeks(numpeople):
#Generate permutations and store pairs in array
    store_pairs = [[]]
    start_permute = np.random.permutation(numpeople)
    store_pairs = AddPairs(start_permute, store_pairs)
    curr_permute = np.random.permutation(numpeople)
    count = 1
    limit = m.factorial(numpeople)/(2**(numpeople/2) * m.factorial(numpeople/2))
    #continue generating until a pair in curr_permute exists in store_pairs
    while CheckExistence(curr_permute, store_pairs) == False:
        store_pairs = AddPairs(curr_permute, store_pairs)
        curr_permute = np.random.permutation(numpeople)
        count += 1
        #sanity check, since there are only n!/(2^(n/2)*(n/2)!) unique pairings
        if count > limit:
            print('impossible scenario!!!')
            break
        if np.mod(count, 1000)==0:
            print('still going, iter:',count)
    return count;

def AddPairs(add_array, master_array):
#add unique pairs to master_array, "10" and "01", "23" and "32" etc.
    sub_array = []
    for i in range(len(add_array)//2):
        sub_array.append(str(add_array[2*i])+str(add_array[2*i+1]))
        sub_array.append(str(add_array[2*i+1])+str(add_array[2*i]))
    #add entire expanded array to the master_array
    master_array.append(sub_array)
    return master_array;

def CheckExistence(check_array, master_array):
#check if the paring in the current permutation exists in the master collection
    test = False
    #check existence for each pair in pairing, if all pairs exist 
    #i.e. count == n/2 then we've got a match
    for j in range(len(master_array)):
        count = 0
        for i in range(len(check_array)//2):
            if np.in1d(str(check_array[2*i])+str(check_array[2*i+1]), master_array[j])[0] == True:
                count = count + 1
        if count == len(check_array)//2:
            test = True
            break  
    return test;

if __name__ == '__main__':
    print('Expected Weeks =', ExpectedWeeks(8))
