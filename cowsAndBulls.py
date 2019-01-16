#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 17:17:54 2019

@author: abhisheklomsh

Create a program that will play the “cows and bulls” game with the user. The game works like this:

Randomly generate a 4-digit number. Ask the user to guess a 4-digit number. 
For every digit that the user guessed correctly in the correct place, they have a “cow”. 
For every digit the user guessed correctly in the wrong place is a “bull.” 
Every time the user makes a guess, tell them how many “cows” and “bulls” they have. 
Once the user guesses the correct number, the game is over. Keep track of the number of guesses the user makes throughout teh game and tell the user at the end.
"""

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
        break
        
    print("You got cow ", cow," and bulls ",bull)
    
    #bull=0
    #cow=0
    #pos_marker=[]