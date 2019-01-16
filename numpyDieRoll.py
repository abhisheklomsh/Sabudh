#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:48:12 2019

@author: abhisheklomsh

Here I tried to solve a challenge given to me where one has to suppose that
A man is standing in front of a staircase and rolls a dice, 
if he gets 1/2/3 he will move one step down
if he gets 4/5 he will move one step upward
if he gets 6 then he will roll die again and move steps upward as number on die
"""
import random
import numpy as np

sampleSize = int(input("Enter value of die rolls"))
import matplotlib.pyplot as plt

def score_cal(score,newValue):   
        if newValue== 1 or newValue== 2 or newValue== 3:
            if score==0:
                pass
            else:
                score -= 1
                
        elif newValue== 4 or newValue== 5:
            score+=1
            
        elif newValue == 6:
            newValue = random.randint(1,6)
            score+=newValue
    
        return score
score = 0
mylist=[]
for i in range(sampleSize+1):
    
    newValue = np.random.choice(6, 1, p=[0.1, 0.1, 0.1, 0.1, 0.3,0.3])+1
    
    mylist.append(newValue)
    score = score_cal(score,newValue)
print("Final score is "+str(score))
#import matplotlib

val1=mylist.count(1)
val2=mylist.count(2)
val3=mylist.count(3)
val4=mylist.count(4)
val5=mylist.count(5)
val6=mylist.count(6)

counted = []
counted.append(val1)
counted.append(val2)
counted.append(val3)
counted.append(val4)
counted.append(val5)
counted.append(val6)
print(counted)

plt.bar(height = counted, left = range(1,7))
plt.show()


    
        


