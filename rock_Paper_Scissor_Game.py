#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:50:27 2019

@author: abhisheklomsh
Make a two-player Rock-Paper-Scissors game. 
(Hint: Ask for player plays (using input), 
compare them, print out a message of congratulations to the winner, 
and ask if the players want to start a new game)

Remember the rules:
    Rock beats scissors
    Scissors beats paper
    Paper beats rock

"""
import random
def playgame(choice):
    all_moves=['rock','paper','scissor']
    comp_choice = random.choice(all_moves)
    print(comp_choice)
    
    if choice == 'rock' and comp_choice == 'scissor':
        print('You won')
        
    elif choice == 'paper' and comp_choice == 'rock':
        print('You won')
        
    elif choice == 'scissor' and comp_choice == 'paper':
        print('You won')
        
    elif choice==comp_choice:
        print("it\'s a draw")
        
    else:
        print("You lose")
        
tryagain = "y"

while tryagain=="y":        
    choice = input("Play your move rock|paper|scissor")
    playgame(choice)
    tryagain=input('wanna try again? press y')