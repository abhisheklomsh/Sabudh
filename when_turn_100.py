
"""
Created on Mon Jan 7

@author: abhisheklomsh

Here we are asking user to enter his age in years and we will tell him in how many years he or she will turn 100
"""

import datetime
now = datetime.datetime.now()


age = int(input('Hi! Please enter your current age in years and press \'Enter\''))
years = 100-age
print('You will turn 100 in either '+str(now.year+years-1)+" "+str(now.year+years))