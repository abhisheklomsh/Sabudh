"""
author: Abhishek Lomsh

Here we ask user to enter a number and in turn we tell him/her if the number is even or odd
"""

num_to_be_checked = input('Hi! Please enter the number to be checked and press \'Enter\'')
even_or_odd = int(num_to_be_checked)%2
if even_or_odd==1:
	print(str(num_to_be_checked)+' is odd')
else:
	print(str(num_to_be_checked)+' is even')