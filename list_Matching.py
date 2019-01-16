#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:45:36 2019

@author: abhisheklomsh

Here I took two lists and printed out common elements
"""

first_list = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
second_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
common=[]
for i in second_list:
    if i in first_list:
        common.append(i)
        
print(common)
        