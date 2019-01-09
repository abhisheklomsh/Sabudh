#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:17:54 2019

@author: abhisheklomsh
"""

import random
import numpy
flag=0
cow=0
bull=0
pos_marker=[]
correct_number=numpy.random.randint(1000,9999)
while cow!=4:
    print(correct_number)
    guess=int(input("Enter your guessed number"))
    if guess== correct_number:
        cow=4
        print("you got ",cow," cows and ",bull," bulls")
    else:
        for iteration in range(4):
            if str(correct_number)[iteration]==str(guess)[iteration]:
                cow+=1
            else:
                pos_marker.append(iteration)
                print(pos_marker)
    
    while len(pos_marker) != flag:
        for iteratn in pos_marker:
            if str(correct_number)[iteratn] in str(guess):
                
                print(str(guess)[iteratn])
                bull+=1
            flag+=1
        
    print("You got cow ", cow," and bulls ",bull)
    bull=0
    cow=0
    pos_marker=[]