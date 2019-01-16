#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:21:54 2019

@author: abhisheklomsh

Generate a random number between 1 and 9 (including 1 and 9). 
Ask the user to guess the number, then tell them whether they guessed too low, too high, or exactly right.
"""
import random
def guessing_game_one(guessed_num):
    correct_num=random.randint(1,9)
    turn=0
    code=''
    while code!='exit':
        if(guessed_num>correct_num):
            print("You guess is too high!")
            
            turn+=1
            code=input("Enter exit to quit or y to try again!")
            if code != 'exit':
                guessed_num=int(input("Enter exit to quit or try again!"))
        elif(guessed_num<correct_num):
            print("You guess is too low!")
            code=input("Enter exit to quit or y to try again!")
            if code != 'exit':
                guessed_num=int(input("Enter exit to quit or try again!"))
            turn+=1
        elif(guessed_num==correct_num):
            print("You guessed it right! \n It took you ",turn," attempt(s)")
            code=input("Enter exit to quit or y to start again!")
            if code != 'exit':
                correct_num=random.randint(1,9)
                turn=0
    if code=='exit':
        print('correct answer was ',correct_num)
    
guessing_game_one(int(input('Enter your guessed number')))
        