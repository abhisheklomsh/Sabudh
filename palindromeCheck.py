#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:51:17 2019

@author: abhisheklomsh
"""
def reverse_word(input_string):
    reversed_word=''
    for i in range(len(input_string)):
        reversed_word+=(input_string[len(input_string)-i-1])
        


input_string = input("Enter a word to check if it is a palindrome")
reversed_word=reverse_word(input_string)
if reversed_word==input_string:
    print("Yes it is a palindrome!")
else:
    print("No it is not a palindome")        
        