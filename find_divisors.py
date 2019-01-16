"""
author: Abhishek Lomsh

Here we ask user to enter a number and we return all divisors of the number
"""
number = int(input('Hi! Please enter the number to find divisors and press \'Enter\''))
mylist = []
if number == 2:
    mylist.append(2)
for divisor in range(2,int(number/2)+1):
    if number%divisor == 0:
        mylist.append(divisor)
    else:
        continue

print("Here are all divisors of "+str(number)+" "+str(mylist))