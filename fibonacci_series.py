#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:27:49 2019

@author: abhisheklomsh
"""
def fibonacci_series(iteration):
    a = 1
    b = 1
    c = 0
    print('Fibonacci series : ')
    print(str(a)+' ')
    
    if(iteration == 2):
        print(str(b)+' ')
        pass
    elif(iteration > 2):
        print(str(b)+' ')
    
    for iterations in range(iteration-2):
        c= a+b
        a=b
        b=c
        print(str(c)+' ')

iteration = int(input("Please enter number of fibonacci series digits to be printed"))
fibonacci_series(iteration)