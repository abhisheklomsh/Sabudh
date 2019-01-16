#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:51:47 2019

@author: abhisheklomsh
Here we ask user to enter a number and we return whether the number is prime or not
"""

def prime_check(input_num):
    flag=1
    if input_num ==1: print(str(input_num)+" is not a prime number")
    else:
        for i in range(2,input_num+1):
            if (input_num==i and flag ==1):
                print(str(input_num)+" is a prime number!")
                pass
            elif(input_num%i==0 and flag==1):
                print("This is not a prime number")
                flag=0
                

prime_check(int(input('Enter number to be cheked if it\'s prime or not')))
    