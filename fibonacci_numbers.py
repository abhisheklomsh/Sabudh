#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:22:14 2019

@author: abhisheklomsh

Here we ask user to input number of fibonacci numbers user wants and we print that many number of fibonacci numbers
"""

def fibonacci_series(iteration):
    a = 1
    b = 1
    c = 0
    print(str(a)+' ')
    print(str(b)+' ')
    if(iteration in [1,2]):
        pass
    else:
        for iterations in range(iteration-2):
            c= a+b
            a=b
            b=c
            print(str(c)+' ')

iteration = int(input("Please enter number of fibonacci series digits to be printed"))
fibonacci_series(iteration)