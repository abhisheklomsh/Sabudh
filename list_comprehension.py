#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:01:28 2019

@author: abhisheklomsh
Here we take in a list from user and store even numbers out of the list 

"""
value = 0
first_list = []
second_list = []
while value != -1:
    value = int(input("Enter a number to be added to list or press -1 to proceed"))
    print(value)
    first_list.append(value)

second_list = [number for number in first_list if number % 2 == 0]

print(second_list)