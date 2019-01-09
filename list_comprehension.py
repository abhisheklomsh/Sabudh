#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:22:14 2019

@author: abhisheklomsh
"""

a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
b = [number for number in a if number % 2 == 0]

print(b)


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